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
    num_swb_long_window = 49
    swb_offset_long_window = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 
                              96, 108, 120, 132, 144, 160, 176, 196, 216, 240, 264, 292, 320, 352, 
                              384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 
                              832, 864, 896, 928, 1024]
    return swb_offset_long_window, num_swb_long_window


if __name__ == "__main__":
    print(get_scalefactor_bands())