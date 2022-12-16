
### NOTE is this where the encoding and decoding commands happen?
### and where things get passed in/held?
## (pg. 16)

### NOTE, ask why soundfile isn't importing correctly

# import soundfile as sf
from scipy.io import wavfile
import numpy as np
import os
from utils.bitarray_utils import BitArray, uint_to_bitarray, bitarray_to_uint
from compressors.advanced_audio_coding.divide_data import creating_blocks
import matplotlib.pyplot as plt

# importing files that contain the functions for the block diagram
from compressors.advanced_audio_coding.filterbank_encode import filterbank_encoder
from compressors.advanced_audio_coding.aac_huffman_coding import aac_huffman_encode, aac_huffman_decode, encode_prob_dist, decode_prob_dist
 
from compressors.advanced_audio_coding.psychoacoustic_model import calculateThresholds, calculateThresholdsOnBlock, expand
from compressors.advanced_audio_coding.scalefactor_bands import get_scalefactor_bands
from compressors.advanced_audio_coding.quantization import forwardQuantizationBlock, inverseQuantizationBlock, quantize, unquantize, quantizeSimple, inverseQuantizeSimple

# importing filterbank decode file
from filterbank_decode import filterbank_decoder

import tensorflow as tf

def compute_mse(original, reconstruction):
    """
    Compute the mean square error.
    """
    return ((original - reconstruction)**2).mean()

def load_wav_audio(filepath):
    '''
    load wav audio as float np array and return the array and the sample rate
    Checks that audio is mono
    '''
    # audio_arr, audio_sr = sf.read(filepath)
    # assert len(audio_arr.shape) == 1 # mono
    # return audio_arr, audio_sr
    audio_sr, audio_arr = wavfile.read(filepath, mmap=False)
    #audio_arr = audio_arr[:,0]
    assert len(audio_arr.shape) == 1 # mono

    # normalize audio
    audio_arr = audio_arr/32768

    return audio_arr, audio_sr

def bit_stream_format_encode(wav_data, fs=44100, channels=1, compression=1):
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
    blocked_data, num_blocks, padded_zeros = creating_blocks(wav_data)

    # add to bitstream the number of blocks, and number of padded zeros at the end
    bitstream += uint_to_bitarray(num_blocks, bit_width=16) # num of blocks can range depending on how long the data is
    bitstream += uint_to_bitarray(padded_zeros, bit_width=11) # num of padded zeros at most will be 2047
    bitstream += uint_to_bitarray(len(wav_data), bit_width=32) # add length of original wav file
    # call the filterbank, return windowed vals and the window shapes (which are assumed to be LONG)
    filtered_data = []
    # initialize prev window to be zeros (since nothing came before)
    prev = np.zeros(2048)
    frame_length = 2048
    halflen = frame_length // 2
    waveform_pad = tf.pad(wav_data.astype(float), [[halflen, 0],])
    filtered_data = np.array(tf.signal.mdct(waveform_pad, frame_length, pad_end=True,
                            window_fn=tf.signal.kaiser_bessel_derived_window))

    # quantization step

    # for i in range(num_blocks):
    #     # call filterbank encode (with default vals: kbd window shape, long seq)
    #     # (since using default values, window_seq = 0, window_shape = 0)
    #     window_seq = 0
    #     window_shape = 0
    #     filtered_data.append(filterbank_encoder(blocked_data[i], i, prev, window_seq, window_shape))
    #     prev = blocked_data[i]
    
    # from filterbank, add window data and window seq to bitstream (1 bit each)
    window_seq = 0
    window_shape = 0
    bitstream += uint_to_bitarray(window_seq, bit_width=1)
    bitstream += uint_to_bitarray(window_shape, bit_width=1)
    # Call the psychoacoustic model and return -------
    thresholds, scaling_not_exp, scaling = calculateThresholdsOnBlock(filtered_data, compression=compression)
    #print(scaling.flatten().tolist())

    # after psychoacoustic, goes into scaling and quantization
    #scaling = scaling.flatten()
    
    quant_spec = quantizeSimple(filtered_data, scaling)

    quant_spec_flattened=quant_spec.flatten()
    data_all = np.concatenate([quant_spec_flattened,scaling_not_exp.flatten()])
    # encode point between the two arrays
    bitstream += uint_to_bitarray(len(quant_spec_flattened), bit_width=32)

    # then huffman coding, which this gets added to bitstream
    # since encoded data is at the end, it can be however long
    encoded_data, prob_dist = aac_huffman_encode(data_all)
    # encode the probability distribution (32 bits for the length of prob dist
    bitstream += encode_prob_dist(prob_dist)
    bitstream += encoded_data
    # bitstream += encode_prob_dist(prob_dist_scaling)
    # bitstream += encoded_scaling

    return bitstream


def bit_stream_format_decode(bitstream):
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
    halflen = 1024
    idx = 16
    fs = bitarray_to_uint(bitstream[:idx])
    channels = bitarray_to_uint(bitstream[idx:idx+1])
    idx += 1
    num_blocks = bitarray_to_uint(bitstream[idx: idx + 16])
    idx += 16
    padded_zeros = bitarray_to_uint(bitstream[idx: idx + 11])
    idx += 11
    n_samples = bitarray_to_uint(bitstream[idx: idx + 32])
    idx += 32
    window_seq = bitarray_to_uint(bitstream[idx:idx+1])
    idx += 1
    window_shape = bitarray_to_uint(bitstream[idx:idx+1])
    idx += 1

    quantization_values_num = bitarray_to_uint(bitstream[idx: idx + 32])
    idx += 32
    # print('encoded probability dist length',len(bitstream[idx:idx+32]))
    prob_dist, num_bits_read = decode_prob_dist(bitstream[idx:])
    idx += num_bits_read

    # Huffman decode
    decoded_data, num_bits_consumed = aac_huffman_decode(bitstream[idx:], prob_dist)
    idx += num_bits_consumed
    #print('decoded_data',decoded_data)
    quant_values = decoded_data.data_list[:quantization_values_num]
    scalings = decoded_data.data_list[quantization_values_num:]
    scalings = expand(np.resize(scalings, (len(scalings)//49, 49)))

    inverse_quant_data = np.resize(np.array(quant_values), (len(quant_values)//halflen, halflen))
    filterbank_coefficients = inverseQuantizeSimple(inverse_quant_data, scalings)
    #inverse_quant_data_unflattened = np.resize(inverse_quant_data, (len(inverse_quant_data)//1024, 1024)) * thr_inv
    # inverse filterbank (filterbank decode)
    # audio_data = []
    # for i in range(num_blocks):
    #     window_sequence = 0
    #     window_shape = 0
    #     fd = filterbank_decoder(inverse_quant_data[i], i, window_sequence, window_shape)
    #     audio_data.append(fd)
    # print(inverse_quant_data_unflattened)
    inverse_mdct = np.array(tf.signal.inverse_mdct(filterbank_coefficients,
                                            window_fn=tf.signal.kaiser_bessel_derived_window))
    frame_length = 2048
    halflen = frame_length // 2
    audio_data = inverse_mdct[halflen:halflen+n_samples]

    # theoretically, should get the audio data back

    return audio_data

def testEndtoEnd(compression=1):
    # import audio file (16 bit wav file)
    audio_arr, audio_sr = load_wav_audio('snippet.wav')

    # Encode audio file
    en_data = bit_stream_format_encode(audio_arr, fs=audio_sr, channels=1,compression=compression)
    print('length of encoded data',len(en_data))

    # Decode audio file
    dec_data = bit_stream_format_decode(en_data)

    dec_data_flattened_int16 = (dec_data*32768).astype(np.int16)
    wavfile.write('compressed_huffman'+str(compression)+'.wav', audio_sr, dec_data_flattened_int16)

    print('MSE', compute_mse(audio_arr,dec_data))
    print('kbps', len(en_data)/1024/(len(audio_arr)/audio_sr))

if __name__ == "__main__":
    testEndtoEnd(1)
    testEndtoEnd(2)
    testEndtoEnd(3)
    testEndtoEnd(4)
    testEndtoEnd(5)
    testEndtoEnd(10)
    #
    #wavfile.write('snippet.wav', audio_sr, audio_arr[2*2694528:2*2694528+441000].astype(np.int16))

    # 441000-1024*430 = 680
    # 1,368

    # 6.453100401880862e-05  c =3

    # 2.4669114549042774e-05 c=1

    # 300746578653449e-3 c=10

    #MSE 8.98347274603052e-05
    #kbps 74.467578125

    #MSE 8.927329020313341e-05
    #kbps 79.6947265625