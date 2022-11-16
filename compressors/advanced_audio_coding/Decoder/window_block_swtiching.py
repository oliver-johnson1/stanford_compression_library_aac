from scipy import signal
import math


def ksb(n,a):
    """
    Inputs: 
        - n: int
        - a: alpha (int)
    Output:
        w_prime: kaiser-bessel kernel window function (pg. 98, but broken reference)
    """
    # if n >= 0 and n <= N/2:
        # w_prime = (math.pi * a * (1 - (n - N/4)/(N/4)))
        ### Maybe scipy signal kaiser works???
    w_prime = signal.kaiser(n,a)
    return w_prime
    # else:
    #     raise("Not within range for ksb")

def get_window_coeffs(window_shape, N, n):
    """
    Inputs: 
        - window_shape: 1 or 0 (1 bit) to indicate which function
        - N: window length
        - n: index?? (int)
    Output:
        - The window coefficient for the function choice

    """

    # a (alpha) is the kernal window alpha factor
    if N == 2048:
        a = 4
    elif N == 256:
        a = 6

    if window_shape == 1:
        if n >= 0 and n <= N/2:
            w_kbd_left_top = 0
            w_kbd_left_bot = 0
            for p in range(n):
                w_kbd_left_top += ksb(p,a)
            for p in range(N/2):
                w_kbd_left_bot += ksb(p,a)
            return math.sqrt(w_kbd_left_top/w_kbd_left_bot)

        elif N/2 >= 0 and n <= N:
            w_kbd_right_top = 0
            w_kbd_right_bot = 0
            for p in range(N-n-1):
                w_kbd_right_top += ksb(p,a)
            for p in range(N/2):
                w_kbd_right_bot += ksb(p,a)
            return math.sqrt(w_kbd_right_top/w_kbd_right_bot)

    elif window_shape == 0:
        #### They are exactly the same for left and right?????????#########
        if n >= 0 and n <= N/2:
            return math.sin(math.pi / N *(n+ 1/2))
        elif N/2 >= 0 and n <= N:
            return math.sin(math.pi / N *(n+ 1/2))

def overlap_add(z_r, z_l,b=False):
    """
    Should overlap and add within the EIGHT_SHORT window seq 
    as well as the first (left) half of every window seq is overlap and add with second (right) half of prev window_seq

    Input:
        - b: bool if EIGHT_SHORT
        - z_l: left
        - z_r: right

    Output:
        - out_i_n: ### Something convoluted if bool is EIGHT_SHORT
        - out_i_n: z_l + z_r (for 0<=n<N/1, N=2048) if bool is not EIGHT_SHORT

    """
    pass

def w_left(n, N, prev_window_shape):
    """
    Takes the previous window shape (0 or 1) to determine whether KBD or Sine
    """
    return get_window_coeffs(prev_window_shape, N, n)

def window(n, window_shape, seq_type):
    """
    Based on the sequence type (ONLY_LONG_SEQ, LONG_START_SEQ, etc.), and n, get the window

    Inputs:
        - n: int
        - window_shape: 0 or 1
        - seq_type: string
    
    Output:
        w: window
    """
    match seq_type:
        case "ONLY_LONG_SEQ":
            if n >= 0 and n < 1024:
                ### takes previous window shape????
                return w_left(n, 2048, prev)
            else:
                return get_window_coeffs(window_shape, 2048, n)
        
        case "LONG_START_SEQ":
            if n >= 0 and n < 1024:
                ### takes previous window shape????
                return w_left(n, 2048, prev)
            elif n >= 1034 and n < 1472:
                return 1.0
            elif n >= 1472 and n < 1600:
                return get_window_coeffs(window_shape, 256, n + 128 -1472)
            else:
                return 0.0
        
        case "EIGHT_SHORT_SEQ":
            # Separate into w_0; w_1 through w_7
            ###### Uhhhh, Need to come back to this, might need to restructure/rethink since it uses each individual w_i for z (pg. 100) [overlap and add between]
            ### For loop to get w_1 through w_7?
            if n >= 0 and n < 128:
                ### prev
                w_0 = w_left(n, 256, prev)
                w_17 = get_window_coeffs(window_shape, 256, n)
            elif n >= 128 and n < 256:
                pass  
        
        case "LONG_STOP_SEQ":
            if n >= 0 and n < 448:
                return 0.0
            elif n >= 448 and n < 576:
                ### prev
                return w_left(n - 448, 256, prev)
            elif n >= 576 and n < 1024:
                return 1.0
            elif n >= 1024 and n < 2048:
                return get_window_coeffs(window_shape, 2048, n)