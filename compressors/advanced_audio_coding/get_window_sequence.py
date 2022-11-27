def get_window_sequence(y:int):
    """
    Gets the window sequence depending on the input of "y"

    Input: 
        - y: int
    Output:
        - N: the length of the window (2048 or 256)
        - seq_type: what type of window sequence (str)
    """
    # set y input (int to binary) to be if ONLY_LONG_SEQ, LONG_START_SEQ, EIGHT_SHORT_SEQ, or LONG_STOP_SEQ


    # match not supported until python 3.10
    if bin(y) == bin(0):#"ONLY_LONG_SEQ":
        return 2048,"ONLY_LONG_SEQ"
    elif bin(y) == bin(1):#"LONG_START_SEQ":
        return 2048, "LONG_START_SEQ"
    elif bin(y) == bin(2):#"EIGHT_SHORT_SEQ":
        return 256, "EIGHT_SHORT_SEQ"
    elif bin(y) == bin(3):#"LONG_STOP_SEQ":
        return 2048, "LONG_STOP_SEQ"

if __name__ == "__main__":
    print(get_window_sequence(1))