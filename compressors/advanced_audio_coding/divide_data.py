
import numpy as np

def creating_blocks(raw_data):
    """
        Dividing up the raw data into 2048 blocks of data

        Input:
            - raw data: np array
        Outputs:
            - block_data: 
                - list of numpy arrays (size 2048-ish)
    """

    N = raw_data.size
    remainder = N%2048
    # if the size of the raw data is not divisible by 2048, pad with zeros
    if remainder != 0:
        raw_data = np.append(raw_data, np.zeros(2048-remainder))
    
    num_blocks = int(raw_data.size / 2048) 

    # split raw data by number of blocks
    block_data = np.split(raw_data, num_blocks)

    return block_data, num_blocks, remainder