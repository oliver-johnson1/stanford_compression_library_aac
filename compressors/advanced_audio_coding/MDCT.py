
import numpy as np
import tensorflow as tf
# Just the equations for computing inverse Discrete Time Cosine Transform (DTCT)
# The psychoacoustic model should perform the fft on the input (can implement here as well)

def forward_filterbank(data, frame_length):
    waveform_pad = tf.pad(data.astype(float), [[frame_length//2, 0],])
    filtered_data = np.array(tf.signal.mdct(waveform_pad, frame_length, pad_end=True,
                            window_fn=tf.signal.kaiser_bessel_derived_window))
    return filtered_data

def reverse_filterbank(filtered_data, frame_length, n_samples):
    inverse_mdct = np.array(tf.signal.inverse_mdct(filtered_data,
                                            window_fn=tf.signal.kaiser_bessel_derived_window))
    recovered_data = inverse_mdct[frame_length//2:frame_length//2+n_samples]
    return recovered_data

def test_filterbank():
    N = 10000
    data = np.random.standard_normal(size=N)
    frame_length = 2048
    filtered_data = forward_filterbank(data,frame_length)
    recovered_data = reverse_filterbank(filtered_data, frame_length, N)
    print(np.allclose(data,recovered_data,rtol=1e-8,atol=1e-8))



def forward_MDCT(k: int, i: int, N: int, z_in: list):
    """
    For encoding

    Inputs: 
        - z_in: windowed input sequence
        - n: sample index
        - i: block index ###NOTE Still need an i (from somewhere)
        - k: spectral coefficient index ###NOTE(are the ks and ns switched in this instance???) Is the ns being generated and ks being passed through
        - N: window length based on the window_seq val
        - n_0: (N/2 + 1)/2 [might just calculate this instead of passing it through]

    Returns:
        - X_i_k = 2 * N sum(z[i][n] * cos(2*pi/N * (n+ n_0) * (k+1/2))) for k=0 to <N/2

    """
    X_i_k = 0

    n_0 = (N/2 + 1)/2
    ns = N-1
    for n in range(ns):
        ### This is if it's the eight sequence
        # X_i_k += (z_in[i][n] * np.cos(2*np.pi/N * (n + n_0) * (k+1/2)))
        X_i_k += (z_in[n] * np.cos(2*np.pi/N * (n + n_0) * (k+1/2)))
    X_i_k *= 2
    return X_i_k


def inverse_MDCT(n: int, i: int, N: int,spec: list):
    """
    For decoding

    Inputs: 
        - n: sample index
        - i: window index ###NOTE need an i from somewhere
        - k: spectral coefficient index (is this just the range for the sum??? Does this need to be passed through???)
        - N: window length based on the window_seq val
        - n_0: (N/2 + 1)/2 [might just calculate this instead of passing it through]

    Returns:
        - x_i_n = 2/N sum(spec[i][k] * cos(n*pi/N * (n+ n_0) * (k+1/2))) for k=0, N/2-1
            ###NOTE(is spec an input???)

    """
    x_i_n = 0
    n_0 = (N/2 + 1)/2
    k_1 = N/2-1
    for k in range(int(k_1)):
        # x_i_n += (spec[i][k] * np.cos(n*np.pi/N * (n + n_0) * (k+1/2)))
        ## above is for eight seq
        x_i_n += (spec[k] * np.cos(n*np.pi/N * (n + n_0) * (k+1/2)))
    x_i_n *= 2/N
    return x_i_n


def testMDCT():
    N = 2048
    x = np.random.normal(size=N)
    X_forward = forward_MDCT(0,0,N,x)
    print(X_forward, len(X_forward))

    X_orig = inverse_MDCT(0,0,N,X_forward)
    print(x)
    print(X_orig)


if __name__ == "__main__":
    test_filterbank()