
### NOTE is this where the encoding and decoding commands happen?
### and where things get passed in/held?
## (pg. 16)

### NOTE, ask why soundfile isn't importing correctly

# import soundfile as sf
from scipy.io import wavfile
import numpy as np
import os
from utils.bitarray_utils import BitArray, uint_to_bitarray
from divide_data import creating_blocks

# importing files that contain the functions for the block diagram
from filterbank_encode import filterbank_encoder
from aac_huffman_coding import aac_huffman_encode, aac_huffman_decode

from psychoacoustic_model import calculateThresholds, calculateThresholdsOnBlock
from scalefactor_bands import get_scalefactor_bands
from quantization import forwardQuantizationBlock, inverseQuantizationBlock

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

    # call the filterbank, return windowed vals and the window shapes (which are assumed to be LONG)
    filtered_data = []
    # initialize prev window to be zeros (since nothing came before)
    prev = np.zeros(2048)
    frame_length = 2048
    halflen = frame_length // 2
    waveform_pad = tf.pad(wav_data.astype(float), [[halflen, 0],])
    filtered_data = tf.signal.mdct(waveform_pad, frame_length, pad_end=True,
                            window_fn=tf.signal.kaiser_bessel_derived_window)


    # for i in range(num_blocks):
    #     # call filterbank encode (with default vals: kbd window shape, long seq)
    #     # (since using default values, window_seq = 0, window_shape = 0)
    #     window_seq = 0
    #     window_shape = 0
    #     filtered_data.append(filterbank_encoder(blocked_data[i], i, prev, window_seq, window_shape))
    #     prev = blocked_data[i]
    
    # from filterbank, add window data and window seq to bitstream (1 bit each)
    # bitstream += uint_to_bitarray(window_seq, bit_width=1)
    # bitstream += uint_to_bitarray(window_shape, bit_width=1)

    # Call the psychoacoustic model and return -------
    thresholds = calculateThresholdsOnBlock(filtered_data)
    #thresholds = [1*np.ones(len(filtered_data[0]))]*len(filtered_data)
    # after psychoacoustic, goes into scaling and quantization    

    quant_spec = forwardQuantizationBlock(filtered_data, thresholds)

    # then huffman coding, which this gets added to bitstream
    # since encoded data is at the end, it can be however long
    # encoded_data = aac_huffman_encode(quant_spec)
    # bitstream += encoded_data

    # return bitstream

    ##########
    # TESTING
    return quant_spec, num_blocks, thresholds
    #########


def bit_stream_format_decode(bitstream, num_blocks, thresholds):
    ################ NOTE: Remove num_blocks and possibly thresholds after testing, since will be in bitstream
    """
    Inputs:
        - the bitstream
            - sample rate (16 bits)
            - channels (1 bit)
            - number of blocks: how many blocks of size 2048 are there (16 bits???)
            - number of padded zeros: at max adding 2047 zeros to the end of a block (11 bits???)
            - the filterbank control info (window seq (default 1 bit (at most 2 bits)) and window shape (1 bit))
            - the noiselssly coded spectra (any number of bits???)

    Outputs:
        - sample rate <- 4-byte int
        - num of channels <- 2-byte int
            - 1 for mono, 2 for stereo, etc
        - num of bits per data sample <- 2-byte int
            - 1 for 8 bits, 2 for 16 bits, etc
        - num of samples in file <- 4-byte int
        - interleaved channel samples
            - (the filterbank window overlapping)
    
    """
    
    ######## FOR TESTING bitstream - quantized data ########
    
    # inverse quantize
    inverse_quant_data = inverseQuantizationBlock(bitstream, thresholds)

    # inverse filterbank (filterbank decode)
    # audio_data = []
    # for i in range(num_blocks):
    #     window_sequence = 0
    #     window_shape = 0
    #     fd = filterbank_decoder(inverse_quant_data[i], i, window_sequence, window_shape)
    #     audio_data.append(fd)

    inverse_mdct = tf.signal.inverse_mdct(inverse_quant_data,
                                            window_fn=tf.signal.kaiser_bessel_derived_window)
    frame_length = 2048
    halflen = frame_length // 2
    audio_data = inverse_mdct[halflen: halflen + 2694528//12]

    # theoretically, should get the audio data back

    return audio_data
    #####################################################



if __name__ == "__main__":
    # pass
    # cur_dir = os.getcwd()
    # filepath = os.path.join(cur_dir,"advanced_audio_coding/original.wav")
    # print(load_wav_audio(filepath))
    audio_arr, audio_sr = load_wav_audio('original.wav')

    data_sample = audio_arr[:2694528//12]
    print('original',data_sample)

    ##### NOTE: Will delete num_blocks and possible thresholds from output
    en_data,num_blocks, thresholds = bit_stream_format_encode(data_sample, fs=44100, channels=1)
    print('length of encoded data',len(en_data))
    dec_data = bit_stream_format_decode(en_data, num_blocks, thresholds)

    print('processed',dec_data)
    dec_data_flattened = np.array(dec_data).flatten()
    print('length',len(dec_data_flattened))
    # thresholds = calculateThresholdsOnBlock(B)
    # Need to normalize
    dec_data_flattened = dec_data_flattened/dec_data_flattened.max()
    
    wavfile.write('compressed.wav', audio_sr, dec_data_flattened)