from compressors.advanced_audio_coding.MDCT import inverse_MDCT
from compressors.advanced_audio_coding.get_window_sequence import get_window_sequence
from compressors.advanced_audio_coding.window_block_swtiching import window, overlap_add  

from compressors.advanced_audio_coding.filterbank_encode import filterbank_encoder
import numpy as np

def filterbank_decoder(spec, i, window_sequence = 0, window_shape = 0):
    """
    Inputs: 
        - inversely quantized spectra (spec???) 
        - filterbank control info (guessing):
                - window_sequence: (2 bits) tell which sequence it is (see get_window_sequence.py for seq types) 
                - window_shape: 0 (KBD) or 1 (Sine) ### However, window shape doesn't change unless I input a new window shape??? (make a class???)
                (from single_channel_element(), channel_pair_element()?)
        -i: window index (### do I pass it in???)
    Output: Time domain reconstructed audio signals
    """

    #### NEED i (window idx) for inverse DTCT

    z_i_n = []
    out_i_n = []
    tracking_window_shapes = []
    i = 0 #window index set to 0 initially

    # window sequence is 2 bits (the bits correspond to the cases)
    N, seq_type = get_window_sequence(window_sequence)

    ### NOTE Do you loop to get n and i???? (probably for n, not sure about i)
    for n in range(N):

        # Something with using the sequence type, and getting the window (using window coeffs), and then returning time domain values (z_i,n)
        ### NOTE In the beginning of loop, no previous window exists, so prev window is zero? maybe???
        prev = 0
        # if i == 0:
        #     prev = 0
        #     i += 1
        # else:
        #     prev = tracking_window_shapes[-1]
        #     i += 1

        # Get the window w
        w = window(n, window_shape, seq_type, prev)

        # Keep track of previous window shape
        tracking_window_shapes.append(window_shape)

        if seq_type != "EIGHT_SHORT_SEQ":
            # n: sample index, i: window index
            x_i_n = inverse_MDCT(n, i, N,spec)

            # Get the z time domain vals
            # z_i_n.append(w * x_i_n)
            z_i_n.append(x_i_n)
        else:
            # Have to do the eight one (in which w should be a list w[0...7])
            if (n >= 0 and n < 448) or (n >= 1600 and n <2048):
                intermediate = 0
            elif n >= 448 and n < 576:
                intermediate = inverse_MDCT(n-448, i, N,spec) * w[0]
            elif n >= 576 and n < 704:
                intermediate = inverse_MDCT(n-448, i, N,spec) * w[0] + inverse_MDCT(n-576, i, N,spec) * w[1]
            elif n >= 704 and n < 832:
                intermediate = inverse_MDCT(n-576, i, N,spec) * w[1] + inverse_MDCT(n-704, i, N,spec) * w[2]
            elif n >= 832 and n < 960:
                intermediate = inverse_MDCT(n-704, i, N,spec) * w[2] + inverse_MDCT(n-832, i, N,spec) * w[3]
            elif n >= 960 and n < 1088:
                intermediate = inverse_MDCT(n-832, i, N,spec) * w[3] + inverse_MDCT(n-960, i, N,spec) * w[4]
            elif n >= 1088 and n < 1216:
                intermediate = inverse_MDCT(n-960, i, N,spec) * w[4] + inverse_MDCT(n-1088, i, N,spec) * w[5]
            elif n >= 1216 and n < 1344:
                intermediate = inverse_MDCT(n-1088, i, N,spec) * w[5] + inverse_MDCT(n-1216, i, N,spec) * w[6]
            elif n >= 1344 and n < 1472:
                intermediate = inverse_MDCT(n-1216, i, N,spec) * w[6] + inverse_MDCT(n-1344, i, N,spec) * w[7]
            elif n >= 1472 and n < 1600:
                intermediate = inverse_MDCT(n-1344, i, N,spec) * w[7]

            z_i_n.append(intermediate)
    ### NOTE overlap and add these z_i_n prev and z_i_n vals (second half of prev and first half of current)
    # if len(z_i_n) > 1:
        # prev = z_i_n[i-1]
        # curr = z_i_n[i]
        # overlapped = overlap_add(prev[len(curr)/2:], curr[:len(curr)/2])
        # out_i_n.append(overlapped)
    # If the beginning of the signal, overlapps current with nothing, so just get current as out_i_n
    # if prev_window_seq is None:
    #     return z_i_n
    # else:
    #     # print(np.shape(prev_window_seq),np.shape(z_i_n))
    #     out_i_n = overlap_add(prev_window_seq[1024:], z_i_n[:1024])
    left = z_i_n[:1024]
    right = z_i_n[1024:]

    # dataL = np.dot(w,left)
    dataL = np.array(w) * left
    # dataR = np.dot(w,right)
    dataR = np.array(w) * right
    out_i_n = np.append(dataL, dataR)

    return out_i_n

if __name__ == "__main__":
    # testing w/o quantizer and scaling
    # just feed back encoder output into decoder
    # testing
    N = 2048
    Fs = 44100
    f = 3000.0
    n = np.arange(N)
    # x = np.cos(2 * np.pi * f * n / Fs) # generated fake signal
    x = np.ones(N)
    print(x)

    seq = 0
    shape = 0
    x_i_k = filterbank_encoder(x, 1, x, seq, shape)
    print(x_i_k)
    # testig decoder
    prev_win_seq = None
    prev_win_seq = filterbank_decoder(x_i_k, 1, window_sequence = 0, window_shape = 0)
    # print(prev_win_seq, 'first filterbank decode')
    print(len(prev_win_seq))
    ### Since decoder relies on previous window sequence to overlap and add, use previous filterbank decode output
    # print(filterbank_decoder(x_i_k, seq, shape,prev_win_seq))
    # second_window = filterbank_decoder(x_i_k, seq, shape,prev_win_seq)
    # print(len(second_window))
    print(prev_win_seq)
    print(x == prev_win_seq)