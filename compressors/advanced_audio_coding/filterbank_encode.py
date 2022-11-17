from DTCT import forward_DTCT
from get_window_sequence import get_window_sequence
from window_block_swtiching import window  



def filterbank(x_i_n, window_sequence, window_shape):
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
        - X_i_k: the transformed signal
        - Some control signal for bitsream formatter ### (window seq and window shape)????
        

    """
    X_i_k = []
    tracking_window_shapes = []
    i = 0 #Block idx, initial is zero
    z_in = []
    prev_x_i_n = x_i_n[:1024]

    # window sequence is 2 bits (the bits correspond to the cases)
    N, seq_type = get_window_sequence(window_sequence)

    ### For k in range N/2 (from MDCT range pg. 170)? loop through spectral coeff idx (does that makes sense???) (no, since k is just in range 0 to N/2)
    for k in range(N/2):
        if i == 0:
            prev = 0
            i += 1
        else:
            prev = tracking_window_shapes[-1]
            i += 1

        # take the x_i_n's and 1024 val of previous window_seq concatenated with 1024 vals of current block (= x_prime)
        # Get the window w (###where does this come in???)
        w = window(k, window_shape, seq_type, prev)

        # Keep track of previous window shape
        tracking_window_shapes.append(window_shape)

        ### concatenate ??
        # x_i_n_prime??? (and then concatenate x_prime into z_in???)
        ### wants 0->1024 of prev, and 1024->2048 of current, but what if it's the EIGHT seq???
        x_i_n_prime = prev_x_i_n.extend(x_i_n[1024:])
        z_in.append(x_i_n_prime)
        prev_x_i_n = x_i_n[1024:]


        X_i_k.append(forward_DTCT(k, i, N, z_in))

    ### return X_i_k?
    return X_i_k
