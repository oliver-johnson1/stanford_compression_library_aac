
### NOTE is this where the encoding and decoding commands happen?
### and where things get passed in/held?
## (pg. 16)

### NOTE, ask why soundfile isn't importing correctly

# import soundfile as sf
from scipy.io import wavfile
import numpy as np
import os
from utils.bitarray_utils import BitArray, uint_to_bitarray, bitarray_to_uint

from divide_data import creating_blocks

# importing files that contain the functions for the block diagram
from filterbank_encode import filterbank_encoder
from aac_huffman_coding import aac_huffman_encode, aac_huffman_decode, encode_prob_dist, decode_prob_dist

from psychoacoustic_model import calculateThresholds, calculateThresholdsOnBlock
from scalefactor_bands import get_scalefactor_bands
from quantization import forwardQuantizationBlock, inverseQuantizationBlock, quantize, unquantize

# importing filterbank decode file
from filterbank_decode import filterbank_decoder

import tensorflow as tf

def load_wav_audio(filepath):
    '''
    load wav audio as float np array and return the array and the sample rate
    Checks that audio is mono
    '''
    # audio_arr, audio_sr = sf.read(filepath)
    # assert len(audio_arr.shape) == 1 # mono
    # return audio_arr, audio_sr
    audio_sr, audio_arr = wavfile.read(filepath, mmap=False)
    assert len(audio_arr.shape) == 1 # mono
    return audio_arr, audio_sr

def bit_stream_format_encode(wav_data, fs=44100, channels=1):
    """
    Inputs (pg. 2):
        - sample rate -> 4-byte int (default 44100 Hz)
        - num of channels -> 2-byte int (default 1, mono)
            - 1 for mono, 2 for stereo, etc
        - wav data: the raw input wav file data

    Outputs:
        - the bitstream
            - sample rate (16 bits)
            - channels (1 bit)
            - number of blocks: how many blocks of size 2048 are there (16 bits???)
            - number of padded zeros: at max adding 2047 zeros to the end of a block (11 bits???)
            - the filterbank control info (window seq (default 1 bit (at most 2 bits)) and window shape (1 bit))
            - threshold (NOTE: currently 1 bit)
            - probability distribution (32 bits)
            - the noiselssly coded spectra (any number of bits???)

    """
    bitstream = uint_to_bitarray(fs, bit_width=16) # sample rate is assumed 44100 Hz

    ### NOTE num channels (assume mono??)
    bitstream += uint_to_bitarray(channels, bit_width=1)

    ## Need to divide up the raw data into 2048 size blocks, then feed into filerbank
    # blocked_data, num_blocks, padded_zeros = creating_blocks(wav_data)
    window_seq = 0
    window_shape = 1
    frame_length = 2048
    halflen = frame_length // 2
    waveform_pad = tf.pad(wav_data.astype(float), [[halflen, 0],])
    # blocked_data, num_blocks, padded_zeros = creating_blocks(waveform_pad)
    # print(padded_zeros,'padded zeros added at the end')
    waveform_pad = waveform_pad.numpy()
    blocked_data = creating_blocks(waveform_pad)

    # add to bitstream the number of blocks, and number of padded zeros at the end
    bitstream += uint_to_bitarray(15, bit_width=16) # num of blocks can range depending on how long the data is
    bitstream += uint_to_bitarray(10, bit_width=11) # num of padded zeros at most will be 2047

    # call the filterbank, return windowed vals and the window shapes (which are assumed to be LONG)
    # waveform_pad = tf.pad(wav_data.astype(float), [[halflen, 0],])
    filtered_data = tf.signal.mdct(waveform_pad, frame_length, pad_end=True,
                            window_fn=tf.signal.kaiser_bessel_derived_window)
    
    # from filterbank, add window data and window seq to bitstream (1 bit each)
    # filtered_data = filterbank_encoder(blocked_data, 
    #                 window_sequence = window_seq, window_shape = window_shape)

    bitstream += uint_to_bitarray(window_seq, bit_width=1)
    bitstream += uint_to_bitarray(window_shape, bit_width=1)

    # Call the psychoacoustic model and return -------
    #thresholds = calculateThresholdsOnBlock(filtered_data)
    
    thresholds = [1*np.ones(1024)]*len(filtered_data)
    # after psychoacoustic, goes into scaling and quantization    
    # print(len(thresholds[0]))


    ############################# NOTE: Oliver, fill in the number of bits you need for the threshold
    bitstream += BitArray('0')
    ################################

    filtered_data = list(np.array(filtered_data).flatten())
    # quant_spec = forwardQuantizationBlock(filtered_data, thresholds)
    quant_spec, (smallest, largest) = quantize(filtered_data,nlevels = 64)

    #print(quant_spec)
    #quant_spec = list(np.array(quant_spec).flatten())
    # then huffman coding, which this gets added to bitstream
    # since encoded data is at the end, it can be however long
    encoded_data, prob_dist = aac_huffman_encode(quant_spec)
    # print('probability dist',prob_dist)

    # encode the probability distribution (32 bits for the length of prob dist and then)
    bitstream += encode_prob_dist(prob_dist)

    bitstream += encoded_data

    return bitstream, smallest, largest


def bit_stream_format_decode(bitstream, smallest, largest):
    """
    Inputs:
        - the bitstream
            - sample rate (16 bits)
            - channels (1 bit)
            - number of blocks: how many blocks of size 2048 are there (16 bits???)
            - number of padded zeros: at max adding 2047 zeros to the end of a block (11 bits???)
            - the filterbank control info (window seq (default 1 bit (at most 2 bits)) and window shape (1 bit))
            - threshold values (NOTE: currently 1 bit, will change)
            - the noiselssly coded spectra (any number of bits???)

    Outputs:
        - the uncompressed audio data
    
    """
    idx = 16
    fs = bitarray_to_uint(bitstream[:idx])
    channels = bitarray_to_uint(bitstream[idx:idx+1])
    idx += 1
    num_blocks = bitarray_to_uint(bitstream[idx: idx + 16])
    idx += 16
    num_zeros = bitarray_to_uint(bitstream[idx: idx + 11])
    idx += 11
    window_seq = bitarray_to_uint(bitstream[idx:idx+1])
    idx += 1
    window_shape = bitarray_to_uint(bitstream[idx:idx+1])
    idx += 1

    #### NOTE: Probably change threshold ###
    thresholds = bitstream[idx:idx+1]
    idx += 1

    # print('encoded probability dist length',len(bitstream[idx:idx+32]))
    prob_dist, num_bits_read = decode_prob_dist(bitstream[idx:])
    print('probability dist num bits read', num_bits_read)
    idx += num_bits_read
    compressed_data = bitstream[idx:]


    # Huffman decode
    decoded_data, num_bits_consumed = aac_huffman_decode(compressed_data, prob_dist)
    #print('decoded_data',decoded_data)
    
    #thresholds = [1*np.ones(1024)]*len(decoded_data[0].data_list)
    # inverse quantize
    #decode_unflatten = decoded_data.resize(len(decoded_data)//1024, 1024)
    #inverse_quant_data = inverseQuantizationBlock(decode_unflatten, thresholds)
 
    inverse_quant_data = unquantize(np.array(decoded_data.data_list), 64, (smallest, largest))
    # print('inverse_quant_data',inverse_quant_data, type(inverse_quant_data))

    inverse_quant_data_unflattened = np.resize(inverse_quant_data, (len(inverse_quant_data)//1024, 1024))
    # inverse filterbank (filterbank decode)
    # print(inverse_quant_data_unflattened)
    print('inverse quant size', inverse_quant_data_unflattened.size)
    inverse_mdct = tf.signal.inverse_mdct(inverse_quant_data_unflattened,
                                            window_fn=tf.signal.kaiser_bessel_derived_window)
    # inverse_mdct = filterbank_decoder(inverse_quant_data_unflattened, 
    #                 window_sequence = window_seq, window_shape = window_shape)
    frame_length = 2048
    halflen = frame_length // 2
    audio_data = inverse_mdct[halflen: halflen + 2694528//12]
    print('length of inverse mdct & zeros', len(inverse_mdct), num_zeros)
    
    # audio_data = inverse_mdct[:-num_zeros]
    

    # theoretically, should get the audio data back

    return audio_data

if __name__ == "__main__":
    audio_arr, audio_sr = load_wav_audio('original.wav')

    data_sample = audio_arr[:2694528//(12)]
    print('original',data_sample)
    print('original len', len(data_sample))

    ##### NOTE: Will delete num_blocks and possible thresholds from output
    en_data, smallest, largest = bit_stream_format_encode(data_sample, fs=44100, channels=1)
    
    print('length of encoded data',len(en_data))
    dec_data = bit_stream_format_decode(en_data, smallest, largest)

    print('processed data',dec_data, len(dec_data))
    dec_data_flattened = np.array(dec_data).flatten()
    print('length of decoded data after flattening',len(dec_data_flattened))
    np.allclose(data_sample, dec_data_flattened, rtol=1e-3, atol=1e-4)

    # thresholds = calculateThresholdsOnBlock(B)
    # Need to normalize
    dec_data_flattened = dec_data_flattened/dec_data_flattened.max()*32768
    wavfile.write('compressed_testing_filter.wav', audio_sr, dec_data_flattened.astype(np.int16))