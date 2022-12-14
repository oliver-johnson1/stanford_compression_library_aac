from bitstream_formatter import load_wav_audio
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio


def get_spectrogram(audio_arr,audio_sr):
    plt.specgram(audio_arr, Fs= audio_sr)
    plt.title(str('Spectrogram of wav file'), 
          fontsize = 14, fontweight ='bold')
    plt.xlabel('Time (Sec)')
    plt.ylabel('Magnitude')
    plt.show()


def sig_noise_ratio(original, compressed):
    pred = torch.Tensor(original)
    target = torch.Tensor(compressed)
    print('Scale invariant signal noise ratio:',
            scale_invariant_signal_noise_ratio(target,pred))



if __name__ == "__main__":
    original = 'original.wav'
    compress = 'compressed_testing_filter_separate_tf.wav'
    original_arr, original_sr = load_wav_audio(original)
    compress_arr, compress_sr = load_wav_audio(original)
    # get_spectrogram(compress_arr, compress_sr)
    sig_noise_ratio(original_arr, compress_arr)
