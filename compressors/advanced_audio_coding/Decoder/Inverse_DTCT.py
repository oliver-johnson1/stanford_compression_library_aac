
import math

# Just the equations for computing inverse Discrete Time Cosine Transform (DTCT)
# The psychoacoustic model should perform the fft on the input (can implement here as well)
# def forward_DTCT():
#     pass 

def inverse_DTCT(n: int,i: int,N: int,spec: list):
    """
    For decoding

    Inputs: 
        - n: sample index
        - i: window index
        - k: spectral coefficient index (is this just the range for the sum??? Does this need to be passed through???)
        - N: window length based on the window_seq val
        - n_0: (N/2 + 1)/2 [might just calculate this instead of passing it through]

    Returns:
        - x_i_n = 2/N sum(spec[i][k] * cos(n*pi/N * (n+ n_0) * (k+1/2))) for k=0, N/2-1
            (is spec an input???)

    """
    x_i_n = 0
    n_0 = (N/2 + 1)/2
    k = N/2-1
    for i in range(k):
        x_i_n += (spec[i][k] * math.cos(n*math.pi/N * (n + n_0) * (k+1/2)))
    x_i_n *= 2/N
    return x_i_n
