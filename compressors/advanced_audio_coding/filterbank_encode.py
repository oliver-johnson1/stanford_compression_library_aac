from MDCT import forward_MDCT
from get_window_sequence import get_window_sequence
from window_block_swtiching import window  

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def filterbank_encoder(x_i_n, i, prev, window_sequence=0, window_shape=0):
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
    X_i_k = []
    tracking_window_shapes = []
    # i = 0 #Block idx, initial is zero
    # z_in = np.zeros(shape=(2,1024))
    prev_x_i_n = prev[:1024]
    prev_window_shape = 0
    print('x_i_n',len(x_i_n))
    # window sequence is 2 bits (the bits correspond to the cases)
    N, seq_type = get_window_sequence(window_sequence)

    ###NOTE For k in range N/2 (from MDCT range pg. 170)? loop through spectral coeff idx (does that makes sense???) (no, since k is just in range 0 to N/2)
    for k in range(int(N/2)):
        # if i == 0:
        #     prev = 0
        #     i += 1
        # else:
        #     prev = tracking_window_shapes[-1]
        #     i += 1

        # take the x_i_n's and 1024 val of previous window_seq concatenated with 1024 vals of current block (= x_prime)
        # Get the window w 
        ### TODO: With the EIGHT seq, need to keep track of "i", 
        # since this is where the block idx matters
        # and thus, has a different method w/ concatentating???
        w = window(k, window_shape, seq_type, prev_window_shape)

        # Keep track of previous window shape
        tracking_window_shapes.append(window_shape)

        ###NOTE concatenate ??
        # x_i_n_prime??? (and then concatenate x_prime into z_in???)
        ###NOTE wants 0->1024 of prev, and 1024->2048 of current, but what if it's the EIGHT seq???
        x_i_n_prime = np.append(prev_x_i_n,x_i_n[1024:])
        # z_in is the windowed input seq
        z_in = x_i_n_prime * w
        # z_in[0] = prev_x_i_n
        # z_in[1] = x_i_n[1024:]
        prev_x_i_n = x_i_n[1024:]

        mdct = forward_MDCT(k, i, N, z_in)
        X_i_k.append(mdct)
    X_i_k = np.array(X_i_k)
    ###NOTE return X_i_k? and bitstream control signal
    return X_i_k

def compare_round_trip():
    samples = 10000
    frame_length = 2048
    # halflen = frame_length // 2
    waveform = tf.random.normal(dtype=tf.float32, shape=[samples])
    print('waveform', len(waveform))
    # waveform_pad = tf.pad(waveform, [[halflen, 0],])
    mdct = tf.signal.mdct(waveform, frame_length, pad_end=False,
                            window_fn=tf.signal.kaiser_bessel_derived_window)
    inverse_mdct = tf.signal.inverse_mdct(mdct,
                                            window_fn=tf.signal.kaiser_bessel_derived_window)
    print('inverse len', len(inverse_mdct))
    # inverse_mdct = inverse_mdct[halflen: halflen + samples]
    return waveform, inverse_mdct


if __name__ == "__main__":
    # testing
    waveform, inverse_mdct = compare_round_trip()
    np.allclose(waveform.numpy(), inverse_mdct.numpy(), rtol=1e-3, atol=1e-4)
    print(waveform-inverse_mdct)
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
