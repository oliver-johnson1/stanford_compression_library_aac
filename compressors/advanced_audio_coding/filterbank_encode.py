from MDCT import forward_MDCT
from get_window_sequence import get_window_sequence
from window_block_swtiching import window  

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy
from divide_data import creating_blocks
from filterbank_decode import filterbank_decoder


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
    prev_window_shape = 1
    N, seq_type = get_window_sequence(window_sequence)
    w = window(N, window_shape, seq_type, prev_window_shape)
    print(np.array(x_i_n).shape)
    print(w.shape)
    x_i_n_blocks = np.multiply(np.array(x_i_n), w)
    print(x_i_n_blocks.shape)

    split_wind = np.split(x_i_n_blocks, 4, axis=-1)
    frame_first = -np.flip(split_wind[2], [-1]) - split_wind[3]
    frame_second = split_wind[0] - np.flip(split_wind[1],[-1])
    frames_first_second = np.concatenate((frame_first, frame_second),
                                         axis=-1)
    # type 4 orthonormal DCT of the real windowed signals in frames_rearranged.
    return scipy.fft.dct(frames_first_second, type=4, n=None, axis=-1, 
            norm=None, overwrite_x=False, workers=None, orthogonalize=None)

def compare_round_trip(waveform):
    print('waveform', len(waveform))
    halflen = len(waveform)//2
    waveform_pad = tf.pad(waveform, [[halflen, 0],])
    mdct = tf.signal.mdct(waveform_pad, frame_length, pad_end=True,
                            window_fn=tf.signal.kaiser_bessel_derived_window)
    print(len(waveform_pad))
    inverse_mdct = tf.signal.inverse_mdct(mdct,
                                            window_fn=tf.signal.kaiser_bessel_derived_window)
    inverse_mdct = inverse_mdct[halflen: halflen + samples]
    print('inverse len', len(inverse_mdct))

    return inverse_mdct, mdct


if __name__ == "__main__":
    # testing
    samples = 10000
    frame_length = 2048
    # halflen = frame_length // 2
    waveform = tf.random.normal(dtype=tf.float32, shape=[samples])

    inverse_mdct, mdct = compare_round_trip(waveform)
    np.allclose(waveform.numpy(), inverse_mdct.numpy(), rtol=1e-3, atol=1e-4)

    blocked_data, num_blocks, padded_zeros = creating_blocks(waveform.numpy())
    print('num blocks', num_blocks)
    testing_encode = filterbank_encoder(blocked_data, window_sequence=0, window_shape=1)
    # np.allclose(mdct, testing_encode, rtol=1e-3, atol=1e-4)

    test_decode = filterbank_decoder(testing_encode, window_sequence = 0, window_shape = 1)
    # remove padded zeros
    print(padded_zeros)
    test_decode = test_decode[:-padded_zeros]
    np.allclose(waveform.numpy(), test_decode, rtol=1e-3, atol=1e-4)
    # N = 2048
    # Fs = 44100
    # f = 3000.0
    # n = np.arange(N)
    # x = np.cos(2 * np.pi * f * n / Fs) # generated fake signal
    # mdcts = tf.signal.mdct(
    #             x,
    #             N,
    #             window_fn=tf.signal.kaiser_bessel_derived_window,
    #             pad_end=False,
    #             norm=None,
    #             name=None
    #         )

    # x_recon = tf.signal.inverse_mdct(
    #             mdcts,
    #             window_fn=tf.signal.kaiser_bessel_derived_window,
    #             norm=None,
    #             name=None
    #         )
    # # seq = 0
    # # shape = 0
    # # x_i_k = filterbank_encoder(x, seq, shape)
    # # print(x_i_k)
    # # print(len(x_i_k[0]))
    # print(x)
    # print(x_recon)
