from scipy import signal
import numpy as np

def kaiser_window(n_2, a):
    """
    Kaiser window, used in kaiser bessel derived window
    Inputs: 
        - n_2: half of n number of points
        - a: alpha (int)
    Output:
        - kaiser bessel window
    
    """
    return signal.windows.kaiser(n_2,a)

def ksb(n,a):
    """
    Kaiser bessel derived window
    Inputs: 
        - n: number of points in the output window (int)
        - a: alpha (int)
    Output:
        w_prime: kaiser-bessel kernel window function (pg. 98, but broken reference)
    """
    halflen = n // 2
    kaiserw = kaiser_window(halflen + 1, a)
    kaiserw_csum = np.cumsum(kaiserw)
    halfw = np.sqrt(kaiserw_csum[:-1] / kaiserw_csum[-1])
    window = np.concatenate((halfw, halfw[::-1]), axis=0)
    return window

    # if n >= 0 and n <= N/2:
        # w_prime = (math.pi * a * (1 - (n - N/4)/(N/4)))
        ### Maybe scipy signal kaiser works???
    # w_prime = signal.windows.kaiser(n,a)
    # return w_prime
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
            return ksb(n, a)
            # w_kbd_left_top = ksb(n,a)
            # w_kbd_left_bot = ksb(N/2,a)
            # # for p in range(n):
            # #     w_kbd_left_top += ksb(p,a)
            # # for p in range(N/2):
            # #     w_kbd_left_bot += ksb(p,a)
            # return np.sqrt(w_kbd_left_top/w_kbd_left_bot)

        elif N/2 >= 0 and n <= N:
            return ksb(n, a)
            # w_kbd_right_top = ksb(N-n-1,a)
            # w_kbd_right_bot = ksb(N/2,a)
            # for p in range(N-n-1):
            #     w_kbd_right_top += ksb(p,a)
            # for p in range(N/2):
            #     w_kbd_right_bot += ksb(p,a)
            # return np.sqrt(w_kbd_right_top/w_kbd_right_bot)

    elif window_shape == 0:
        #### They are exactly the same for left and right?????????#########
        # if n >= 0 and n <= N/2:
        return np.sin(np.pi / N *(n + 1/2))
        # elif N/2 >= 0 and n <= N:
            # return math.sin(math.pi / N *(n + 1/2))

def overlap_add(signal, overlap_step):
    """
    Almost the same overlap and add within the EIGHT_SHORT window seq 
    First (left) half of every window seq is overlap and add with second (right) half of prev window_seq

    Input:
        - signal: the overlapped reconstructed signal
        - overlap_step: the offset where the signal is overlapped

    Output:
        - out_i_n:  (for 0<=n<N/1, N=2048) if bool is not EIGHT_SHORT

    """
    outer_dim = signal.shape[:-2]
    print('shapes for overlap add', signal.shape,len(outer_dim), overlap_step)
    frame_len = signal.shape[1]
    frames = signal.shape[0]

    output_len = (frame_len //2 + overlap_step) * (frames)
    print('expected output len', output_len)

    output_shape = np.concatenate([outer_dim, [output_len]],0)
    print(output_shape)
    return np.reshape(signal, int(output_shape))
    # if same length, can just add for each index
    # if len(z_r) == len(z_l):
    #     return list(np.add(z_l, z_r))
    ###NOTE else, not same length for some reason, raise error?


def w_left(n, N, prev_window_shape):
    """
    Takes the previous window shape (0 or 1) to determine whether KBD or Sine
    """
    return get_window_coeffs(prev_window_shape, N, n)

def window(n:int, window_shape, seq_type:str, prev = 1):
    """
    Based on the sequence type (ONLY_LONG_SEQ, LONG_START_SEQ, etc.), and n, get the window

    Inputs:
        - n: int
        - window_shape: 0 or 1
        - seq_type: string
        - prev: previous window shape (still 0 or 1)
    
    Output:
        w: window
    """
    # Apparently, Python <= 3.9 doesn't support case matching
    if seq_type == "ONLY_LONG_SEQ":
        if n >= 0 and n < 1024:
            ### takes previous window shape????
            return w_left(n, 2048, prev)
        else:
            return get_window_coeffs(window_shape, 2048, n)
        
    elif seq_type == "LONG_START_SEQ":
        if n >= 0 and n < 1024:
            ### takes previous window shape????
            return w_left(n, 2048, prev)
        elif n >= 1034 and n < 1472:
            return 1.0
        elif n >= 1472 and n < 1600:
            return get_window_coeffs(window_shape, 256, n + 128 -1472)
        else:
            return 0.0
        
    elif seq_type == "EIGHT_SHORT_SEQ":
        # Separate into w_0; w_1 through w_7
        ### For loop to get w_1 through w_7?
        w = []
        if n >= 0 and n < 128:
            # w_0 case differs
            w.append(w_left(n, 256, prev))
        elif n >= 128 and n < 256:
            # w_0
            w.append(get_window_coeffs(window_shape, 256, n))

        # loop through to get w_1 to w_7, the window_shape and n should take care of which coeffs it is
        for _ in range(7):
            w.append(get_window_coeffs(window_shape, 256, n))
        return w  
                
        
    elif seq_type == "LONG_STOP_SEQ":
        if n >= 0 and n < 448:
            return 0.0
        elif n >= 448 and n < 576:
            return w_left(n - 448, 256, prev)
        elif n >= 576 and n < 1024:
            return 1.0
        elif n >= 1024 and n < 2048:
            return get_window_coeffs(window_shape, 2048, n)

def testing_window(N,x):
    """ Testing window types (KSB and sine)"""

    signalSineWindowed = np.zeros(N)
    # signalKSBWindowed = np.zeros(N)
    for i in range(len(x)):
        signalSineWindowed[i] = get_window_coeffs(0, N, i)
    signalKSBWindowed = ksb(len(x), 4) #alpha = 4, for testing
    print(signalSineWindowed, 'sine, not multiplied by x')
    print(x*signalSineWindowed, 'sine, mult by x')

    print(signalKSBWindowed, 'KSB, not multiplied by x')
    print(x*signalKSBWindowed, 'KSB, mult by x')

def test_overlap(z_r,z_l):
    """ Testing overlap and add"""
    # z_r = [1,2,3]
    # z_l = [2,4,5]
    overlapped = overlap_add(z_r, z_l)
    print(len(overlapped), 'overlapped and added')
    # Check if when you subtract the right, you get back the left
    assert (list(np.subtract(overlapped,z_r)) == z_l).all()

if __name__ == "__main__":

    N = 1024
    Fs = 44100
    f = 3000.0
    n = np.arange(N)
    x = np.cos(2 * np.pi * f * n / Fs) # generated fake signal

    f2 = 2000.0
    x2 = np.cos(2 * np.pi * f * n / Fs)

    testing_window(N,x)
    test_overlap(x2,x)