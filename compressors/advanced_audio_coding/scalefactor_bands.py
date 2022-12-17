import numpy as np

def get_scalefactor_bands():
    """
    swb_offset_long_window[swb]
    Table containing the index of the lowest spectral coefficient of scalefactor band sfb for long windows. 
    This Table has to be selected depending on the sampling frequency. See subclause 8.9.
    
    num_swb_long_window
    number of scalefactor bands for long windows. 
    This number has to be selected depending on the sampling frequency. See subclause 8.9.

    win: window index within group.
    sfb: scalefactor band index within group.
    swb: scalefactor window band index within window.

    """
    # LONG_WINDOW, LONG_START_WINDOW, LONG_STOP_WINDOW at 44.1 kHz and 48 kHz
    # Our sample wave file has a sample rate of 44100 Hz
    # Table 45
    num_swb_long_window = 49
    swb_offset_long_window = np.array([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 
                                       96, 108, 120, 132, 144, 160, 176, 196, 216, 240, 264, 292, 320, 352, 
                                       384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 
                                       832, 864, 896, 928, 1024])
    return swb_offset_long_window, num_swb_long_window


def get_theshold_in_quiet():
    """
    Refines threshold in quiet values based on table C.15 in ISO/IEC STANDARD 13818-7
    (Information technology - Generic coding of moving pictures and associated audio 
    information - Part 7: Advanced Audio Coding (AAC))
    """
    qsthr = np.array([40.29,35.29,35.29,32.29,27.29,25.29,25.29,25.29,25.29,27.05,27.05,27.05,27.05,28.3,
                      28.3,28.3,29.27,30.06,30.73,30.73,31.31,31.82,32.28,33.07,33.42,34.04,34.32,34.83,
                      35.29,38.89,39.08,41.75,42.05,47.46,47.84,48.19,53.4,58.61,59,59.52,64.75,69.98,
                      69.98,70.54,70.54,71.08,71.08,71.72,72.09])
    # Needs to be scaled by lsb
    qsthr_scaled = 2.2204e-16 * 1024/2 * 10**(qsthr/10)
    return qsthr_scaled
if __name__ == "__main__":
    print(get_scalefactor_bands())
    print(get_theshold_in_quiet())