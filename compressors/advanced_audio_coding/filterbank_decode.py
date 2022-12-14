from MDCT import inverse_MDCT
from get_window_sequence import get_window_sequence
from window_block_swtiching import window, overlap_add  

# from filterbank_encode import filterbank_encoder
import numpy as np
import scipy

def filterbank_decoder(spec, window_sequence = 0, window_shape = 1):
    """
    Inputs: 
        - inversely quantized spectra (spec???) 
        - filterbank control info (guessing):
                - window_sequence: (2 bits) tell which sequence it is (see get_window_sequence.py for seq types) 
                - window_shape: 0 (KBD) or 1 (Sine) ### However, window shape doesn't change unless I input a new window shape??? (make a class???)
                (from single_channel_element(), channel_pair_element()?)
        -i: window index (### do I pass it in???)
    Output: Time domain reconstructed audio signals
    """

    half_len = spec.shape[-1] 
    # idct2 = (0.5 / half_len) * scipy.fft.dct(spec, type=2, n=2048, axis=-1, 
    #         norm=None, overwrite_x=False, workers=None, orthogonalize=None)
    idct2 = scipy.fft.idct(spec, type=4, n=None, axis=-1, 
            norm=None, overwrite_x=False, workers=None, orthogonalize=None)
    # idct2 = (0.5 / half_len) * inverse_MDCT(2048,spec)
    # idct4 = idct2[..., 1::2]

    split_result = np.split(idct2, 2, axis=-1)
    print(split_result[1].shape,'split result 1 shape',idct2.shape)
    reconstruct = np.concatenate((split_result[1],
                                    -np.fliplr(split_result[1]),
                                    -np.fliplr(split_result[0]),
                                    -split_result[0]), axis=-1)
    print(reconstruct.shape,'reconstructed shape pre window')                                    
    N, seq = get_window_sequence(window_sequence)
    # w = window(2 * half_len, window_shape, seq, prev = 1)
    # print('decode window', w.size, half_len)
    # reconstruct = reconstruct * w
    #reconstruct *= 1.0 / np.sqrt(2)
    print(reconstruct.shape,'reconstructed shape')
    reconstruct_reshaped = overlap_add(reconstruct, half_len) 
    # reconstruct_reshaped = reconstruct.flatten()
    print(reconstruct_reshaped.shape,'reconstructed reshaped shape')
    return reconstruct_reshaped

if __name__ == "__main__":
    # testing w/o quantizer and scaling
    # just feed back encoder output into decoder
    # testing
    pass
    # N = 2048
    # Fs = 44100
    # f = 3000.0
    # n = np.arange(N)
    # # x = np.cos(2 * np.pi * f * n / Fs) # generated fake signal
    # x = np.ones(N)
    # print(x)

    # seq = 0
    # shape = 0
    # x_i_k = filterbank_encoder(x, 1, x, seq, shape)
    # print(x_i_k)
    # # testig decoder
    # prev_win_seq = None
    # prev_win_seq = filterbank_decoder(x_i_k, 1, window_sequence = 0, window_shape = 0)
    # # print(prev_win_seq, 'first filterbank decode')
    # print(len(prev_win_seq))
    # ### Since decoder relies on previous window sequence to overlap and add, use previous filterbank decode output
    # # print(filterbank_decoder(x_i_k, seq, shape,prev_win_seq))
    # # second_window = filterbank_decoder(x_i_k, seq, shape,prev_win_seq)
    # # print(len(second_window))
    # print(prev_win_seq)
    # print(x == prev_win_seq)