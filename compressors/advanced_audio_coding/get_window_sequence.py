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