
### NOTE is this where the encoding and decoding commands happen?
### and where things get passed in/held?
## (pg. 16)

### NOTE, ask why soundfile isn't importing correctly

# import soundfile as sf
from scipy.io import wavfile
import os
from utils.bitarray_utils import BitArray, uint_to_bitarray

def load_wav_audio(filepath):
    '''
    load wav audio as float np array and return the array and the sample rate
    Checks that audio is mono
    '''
    # audio_arr, audio_sr = sf.read(filepath)
    # assert len(audio_arr.shape) == 1 # mono
    # return audio_arr, audio_sr
    audio_sr, audio_arr = wavfile.read(filepath, mmap=False)
    assert len(audio_arr.shape) == 1 # mono
    return audio_arr, audio_sr

def bit_stream_format_encode(Fs, num_channel, bits_per_sample, num_sample, inter_channel):
    """
    Inputs (pg. 2):
        - sample rate -> 4-byte int
        - num of channels -> 2-byte int
            - 1 for mono, 2 for stereo, etc
        - num of bits per data sample -> 2-byte int
            - 1 for 8 bits, 2 for 16 bits, etc
        - num of samples in file -> 4-byte int
        - interleaved channel samples
            - (the filterbank window overlapping, noiseless coding signal)

    Outputs:
        - the bitstream
            - the filterbank control info
            - the sectioning info for the noiselessly coded spectra
            - the noiselssly coded spectra

    """
    # bitstream = BitArray('1')

    # if fs less than or equal to 4 byte int
    if Fs <= 2**31-1:
        fs = uint_to_bitarray(Fs)
    else:
        fs = uint_to_bitarray(2**32)
    bitstream = fs

    ### NOTE num channels (assume mono??)
    bitstream += BitArray('1')

    # num of bits per data sample
    ### ???????? where do we get that from, filterbank???? 
    bitstream += pcm()

    # huffman encoded noiseleslly coded spectra
    # HCB (huffman coding book), sect_cb (section of codebook from 0 to 16)
    #sect_sfb_offset: pg. 46
    ZERO_HCB = 0
    ESC_HCB = 16
    # Syntax of spectral data (pg. 25):
    for g in num_window_groups:
        for i in num_sec[g]:
            if sect_cb[g][i] != ZERO_HCB and sect_cb[g][i] <= ESC_HCB:
                for k in range(sect_sfb_offset[g][sect_start[g][i]], sect_sfb_offset[g][sect_end[g][i]]):
                    if sect_cb[h][i] < FIRST_PAIR_HCB:
                        bitstream += hcod[sect_cb[g][i]][w][x][y][z] # bits 1-16 (pg. 71)



    return bitstream


def bit_stream_format_decode(bitstream):
    """
    Inputs:
        - the bitstream
            - the sectioning info
            - the noislessly coded spectra

    Outputs:
        - sample rate <- 4-byte int
        - num of channels <- 2-byte int
            - 1 for mono, 2 for stereo, etc
        - num of bits per data sample <- 2-byte int
            - 1 for 8 bits, 2 for 16 bits, etc
        - num of samples in file <- 4-byte int
        - interleaved channel samples
            - (the filterbank window overlapping)
    
    """
    pass


if __name__ == "__main__":
    pass
    # cur_dir = os.getcwd()
    # filepath = os.path.join(cur_dir,"advanced_audio_coding/original.wav")
    # print(load_wav_audio(filepath))