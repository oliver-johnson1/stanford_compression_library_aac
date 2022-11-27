
### NOTE is this where the encoding and decoding commands happen?
### and where things get passed in/held?
## (pg. 16)

### NOTE, ask why soundfile is importing correctly

import soundfile as sf
import os

def load_wav_audio(filepath):
    '''
    load wav audio as float np array and return the array and the sample rate
    Checks that audio is mono
    '''
    audio_arr, audio_sr = sf.read(filepath)
    assert len(audio_arr.shape) == 1 # mono
    return audio_arr, audio_sr



if __name__ == "__main__":
    pass
    # cur_dir = os.getcwd()
    # filepath = os.path.join(cur_dir,"input_abba_sound.wav")
    # filepath = 'input_abba_sound.wav'
    # load_wav_audio(filepath)