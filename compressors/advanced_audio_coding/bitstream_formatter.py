
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
            - (the filterbank window overlapping)

    Outputs:
        - the bitstream
            - the filterbank control info
            - the sectioning info for the noiselessly coded spectra
            - the noiselssly coded spectra

    """
    bitstream = BitArray('1')

    # if fs less than or equal to 4 byte int
    if Fs <= 2**31-1:
        fs = uint_to_bitarray(Fs)
    else:
        fs = uint_to_bitarray(2**32)
    bitstream += fs

    ### NOTE num channels (assume mono??)
    bitstream += BitArray('1')

    # num of bits per data sample
    ### ???????? where do we get that from, filterbank????


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
    # filepath = os.path.join(cur_dir,"original.wav")
    # filepath = 'original.wav'
    # load_wav_audio(filepath)