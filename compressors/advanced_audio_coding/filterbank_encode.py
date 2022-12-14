from MDCT import forward_MDCT
from get_window_sequence import get_window_sequence
from window_block_swtiching import window  

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy
from divide_data import creating_blocks
from filterbank_decode import filterbank_decoder

from scipy.io import wavfile

def filterbank_encoder(x_i_n, window_sequence=0, window_shape=1):
    """
    Forward filterbank to encode the sequence

        Takes the appropriate block of time samples, modulates them by an
        appropriate window function, then performs MDCT.

        Each block of input samples is overalpped by 50% with the immediately
        preciding block and the current block.

    Inputs:
        - x_i_n: the 2048 time-domain values to be windowed
        - window_sequence: (2 bits) tell which sequence it is (see get_window_sequence.py for seq types) 
        - window_shape: 0 (KBD) or 1 (Sine)
        - i: block idx (##is this passed in???)
    
    Output:
        - X_i_k: the transformed signal (np array)
        - Some control signal for bitsream formatter ###NOTE (window seq and window shape or spectral data)????
        

    """
    ##### Something with frames and how tf handles them
    prev_window_shape = 1
    N, seq_type = get_window_sequence(window_sequence)
    # w = window(N, window_shape, seq_type, prev_window_shape)
    # print(np.array(x_i_n).shape,'block shape')
    # print(w.shape,'window shape')
    #### x_i_n_blocks = np.multiply(np.array(x_i_n), w)
    # x_i_n_blocks = x_i_n * w
    x_i_n_blocks = x_i_n * 1.0 / np.sqrt(2) ### testing if window 
    print(x_i_n_blocks.shape,'window block shape')


    split_wind = np.split(x_i_n_blocks, 4, axis=-1)
    print(len(split_wind),'window split shape')
    frame_first = -np.fliplr(split_wind[2]) - split_wind[3]
    print(frame_first.shape,'first frame shape')
    frame_second = split_wind[0] - np.fliplr(split_wind[1])
    frames_first_second = np.concatenate((frame_first, frame_second), axis=-1)
    print(frames_first_second.shape,'concatenated frames shape')
    # type 4 orthonormal DCT
    # dct2 = scipy.fft.dct(frames_first_second, type=2, n=2048, axis=-1, 
    #         norm=None, overwrite_x=False, workers=None, orthogonalize=None)
    dct2 = forward_MDCT(N, frames_first_second)
    dct4 = dct2[..., 1::2]
    return dct2

def compare_round_trip(waveform):
    print('waveform', len(waveform))
    halflen = len(waveform)//2
    waveform_pad = tf.pad(waveform.astype(float), [[halflen, 0],])
    print(waveform_pad.shape, 'waveform padding shape')
    mdct = tf.signal.mdct(waveform_pad, frame_length, pad_end=True,
                            window_fn=tf.signal.kaiser_bessel_derived_window)
    print(mdct.shape,'len of mdct')
    inverse_mdct = tf.signal.inverse_mdct(mdct,
                                            window_fn=tf.signal.kaiser_bessel_derived_window)
    inverse_mdct = inverse_mdct[halflen: halflen + len(waveform)]
    print('inverse len', len(inverse_mdct))

    return inverse_mdct, mdct


if __name__ == "__main__":
    # testing
    # samples = 20000
    frame_length = 2048
    # # halflen = frame_length // 2
    # waveform = tf.random.normal(dtype=tf.float32, shape=[samples])
    audio_sr, audio_arr = wavfile.read('original.wav', mmap=False)

    data_sample = audio_arr[:2694528//(12)]
    waveform = data_sample
    print('original',data_sample)
    print('original len', len(data_sample))

    inverse_mdct, mdct = compare_round_trip(waveform)
    np.allclose(waveform, inverse_mdct.numpy(), rtol=1e-3, atol=1e-4)

    halflen = len(waveform)//2
    waveform_pad = tf.pad(waveform, [[halflen, 0],])
    waveform_pad = waveform_pad.numpy()
    blocked_data = creating_blocks(waveform_pad)
    # blocked_data, num_blocks, padded_zeros = creating_blocks(waveform_pad)
    # blocked_data, num_blocks, padded_zeros = creating_blocks(waveform.numpy())
    # print('num blocks', num_blocks,'num of added zeros:', padded_zeros)
    testing_encode = filterbank_encoder(blocked_data, window_sequence=0, window_shape=1)
    # np.allclose(mdct, testing_encode, rtol=1e-3, atol=1e-4)
    print(testing_encode.shape,'shape of the encoding')

    test_decode = filterbank_decoder(testing_encode, window_sequence = 0, window_shape = 1)
    frame_length = 2048
    halflen = frame_length // 2
    test_decode = abs(test_decode[halflen: halflen + len(waveform)])
    # remove padded zeros
    # print(padded_zeros)
    # test_decode = test_decode[:-padded_zeros]
    print(waveform, test_decode)
    assert True == np.allclose(waveform, test_decode, rtol=1e-5, atol=1e-6)
    assert True == np.allclose(inverse_mdct.numpy(), test_decode, rtol=1e-5, atol=1e-6)
    wavfile.write('compressed_testing_filter_separate_testing_2.wav', audio_sr, test_decode.astype(np.int16))
    wavfile.write('orig_new.wav', audio_sr, waveform.astype(np.int16))
