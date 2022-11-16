from Decoder.Inverse_DTCT import inverse_DTCT
from Decoder.window_block_swtiching import window, overlap_add

# Should I make it a class, so I can share N across funcs? (Do I need to?)

# def single_channel_element():
#     return individual_channel_stream()
       

def get_window_sequence(y):
    # set y input (int to binary) to be if ONLY_LONG_SEQ, LONG_START_SEQ, EIGHT_SHORT_SEQ, or LONG_STOP_SEQ
    match bin(y):
        case bin(0):#"ONLY_LONG_SEQ":
            return 2048,"ONLY_LONG_SEQ"
        case bin(1):#"LONG_START_SEQ":
            return 2048, "LONG_START_SEQ"
        case bin(2):#"EIGHT_SHORT_SEQ":
            return 256, "EIGHT_SHORT_SEQ"
        case bin(3):#"LONG_STOP_SEQ":
            return 2048, "LONG_STOP_SEQ"

def filterbank(spec, window_sequence, window_shape):
    """
    Inputs: 
        - inversely quantized spectra (spec???) 
        - filterbank control info:
                - window_sequence, 
                - window_shape
                (from single_channel_element(), channel_pair_element()?)
    Output: Time domain reconstructed audio signals
    """
    z_i_n = []
    out_i_n = []
    tracking_window_shapes = []

    # window sequence is 2 bits (the bits correspond to the cases)
    N, seq_type = get_window_sequence(window_sequence)

    ### Do you loop to get n and i???? (probably for n, not sure about i)
    for n in range(N):

        # Something with using the sequence type, and getting the window (using window coeffs), and then returning time domain values (z_i,n)
        ### In the beginning of loop, no previous window exists, so prev window is zero? maybe???
        if tracking_window_shapes is []:
            prev = 0
        else:
            prev = tracking_window_shapes[-1]

        # Get the window w
        w = window(n, window_shape, seq_type, prev)

        # Keep track of previous window shape
        tracking_window_shapes.append(window_shape)

        if seq_type != "EIGHT_SHORT_SEQ":
            # n: sample index, i: window index
            x_i_n = inverse_DTCT(n,N,spec)

            # Get the z time domain vals
            z_i_n.append(w * x_i_n)
        else:
            # Have to do the eight one (in which w should be a list w[0...7])
            if (n >= 0 and n < 448) or (n >= 1600 and n <2048):
                intermediate = 0
            elif n >= 448 and n < 576:
                intermediate = inverse_DTCT(n-448,N,spec) * w[0]
            elif n >= 576 and n < 704:
                intermediate = inverse_DTCT(n-448,N,spec) * w[0] + inverse_DTCT(n-576,N,spec) * w[1]
            elif n >= 704 and n < 832:
                intermediate = inverse_DTCT(n-576,N,spec) * w[1] + inverse_DTCT(n-704,N,spec) * w[2]
            elif n >= 832 and n < 960:
                intermediate = inverse_DTCT(n-704,N,spec) * w[2] + inverse_DTCT(n-832,N,spec) * w[3]
            elif n >= 960 and n < 1088:
                intermediate = inverse_DTCT(n-832,N,spec) * w[3] + inverse_DTCT(n-960,N,spec) * w[4]
            elif n >= 1088 and n < 1216:
                intermediate = inverse_DTCT(n-960,N,spec) * w[4] + inverse_DTCT(n-1088,N,spec) * w[5]
            elif n >= 1216 and n < 1344:
                intermediate = inverse_DTCT(n-1088,N,spec) * w[5] + inverse_DTCT(n-1216,N,spec) * w[6]
            elif n >= 1344 and n < 1472:
                intermediate = inverse_DTCT(n-1216,N,spec) * w[6] + inverse_DTCT(n-1344,N,spec) * w[7]
            elif n >= 1472 and n < 1600:
                intermediate = inverse_DTCT(n-1344,N,spec) * w[7]

            z_i_n.append(intermediate)
        ### overlap and add these z_i_n prev and z_i_n vals
        if len(z_i_n) > 1:
            out_i_n.append(overlap_add(z_i_n[-2], z_i_n[-1]))
            # # resetting z_i_n (technically don't have to, so might delete)
            # z_i_n = []
    return out_i_n