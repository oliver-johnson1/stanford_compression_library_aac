from Decoder.Inverse_DTCT import inverse_DTCT
from Decoder.window_block_swtiching import window

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

    # window sequence is 2 bits (the bits correspond to the cases)
    N, seq_type = get_window_sequence(window_sequence)

    ### Do you loop to get n and i????

    # n: sample index, i: window index
    x_i_n = inverse_DTCT(n,i,N,spec)


    # Something with using the sequence type, and getting the window (using window coeffs), and then returning time domain values (z_i,n)
    w = window(n, window_shape, seq_type)
    z_i_n = w * x_i_n

    ### Then return these z values????