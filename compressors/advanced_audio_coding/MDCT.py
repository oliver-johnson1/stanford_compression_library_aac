
import numpy as np
import math
import scipy
import cmath

# Just the equations for computing inverse Discrete Time Cosine Transform (DTCT)
# The psychoacoustic model should perform the fft on the input (can implement here as well)
def forward_MDCT(N: int, z_in):
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
    # axis_dim = z_in.shape[-1]
    # axis_dim_float = float(axis_dim)
    # scale = 2.0 * cmath.exp(complex(np.array([complex(0, -i * cmath.pi * 0.5 /axis_dim_float) 
    #                         for i in range(axis_dim)]).sum()))
    # dct2 = np.real(
    #       scipy.fft.rfft(z_in, n=2 * axis_dim)[..., :axis_dim] * scale)
    dct2 = scipy.fft.dct(z_in, type=4, n=None, axis=-1, 
            norm=None, overwrite_x=False, workers=None, orthogonalize=None)
    return dct2              
    # X_i_k = 0

    # n_0 = (N/2 + 1)/2
    # ns = N-1
    # for n in range(ns):
    #     ### This is if it's the eight sequence
    #     # X_i_k += (z_in[i][n] * np.cos(2*np.pi/N * (n + n_0) * (k+1/2)))
    #     X_i_k += (z_in[n] * np.cos(2*np.pi/N * (n + n_0) * (k+1/2)))
    # X_i_k *= 2
    # return X_i_k


def inverse_MDCT(N: int,spec):
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
    # return forward_MDCT(N, spec)
    half_len = spec.shape[-1]
    idct2 = scipy.fft.idct(spec, type=4, n=None, axis=-1, 
            norm=None, overwrite_x=False, workers=None, orthogonalize=None)
    return idct2
    # x_i_n = 0
    # n_0 = (N/2 + 1)/2
    # k_1 = N/2-1
    # for k in range(int(k_1)):
    #     # x_i_n += (spec[i][k] * np.cos(n*np.pi/N * (n + n_0) * (k+1/2)))
    #     ## above is for eight seq
    #     x_i_n += (spec[k] * np.cos(n*np.pi/N * (n + n_0) * (k+1/2)))
    # x_i_n *= 2/N
    # return x_i_n


def testMDCT():
    N = 2048
    x = np.random.normal(size=N)
    X_forward = forward_MDCT(N,x)
    print(X_forward, len(X_forward))

    X_orig = inverse_MDCT(N,X_forward)
    print(x)
    print(X_orig)

    assert np.allclose(x, X_orig,rtol=1e-3, atol=1e-4)


if __name__ == "__main__":
    testMDCT()