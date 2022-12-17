import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt


def compute_mse(original, reconstruction):
    """
    Compute the mean square error.
    """
    return ((original - reconstruction)**2).mean()

def get_spectrogram(title, audio_arr, audio_sr):
    """
    Generates a spectrogram using pyplot from the given audio array

    Saves these spectrograms as pngs
    """
    plt.specgram(audio_arr, Fs= audio_sr)
    plt.xlabel('Time (Sec)')
    plt.ylabel('Frequency')
    plt.savefig(str(title + '.png'), dpi=250)

def load_wav_audio(filepath):
    '''
    load wav audio as float np array and return the array and the sample rate
    Checks that audio is mono
    '''
    # get the audio sample rate and audio array 
    audio_sr, audio_arr = wavfile.read(filepath, mmap=False)
    
    assert len(audio_arr.shape) == 1 # check to make sure it's mono

    # finds the scaling factor based on the data type of the np arr
    if(audio_arr.dtype == np.int16):
        scaling_factor = 32768
    elif(audio_arr.dtype == np.int32):
        scaling_factor = 2147483648
    elif(audio_arr.dtype == np.float32):
        scaling_factor = 1.0
    else:
        raise("Data type not supported: must be int16, int32 or float32")

    # normalize audio (can assume 16 bit)
    audio_arr = audio_arr/scaling_factor
    return audio_arr, audio_sr

def write_wav_audio(filename, data, sr):
    """
    Write wav audio
    Only 16 bit is supported
    data should be in range [-1,1) and is scaled the same amount as the loda file
    """
    data_int16 = (data*32768).astype(np.int16)
    wavfile.write(filename, sr, data_int16)