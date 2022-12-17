import numpy as np
from utils.bitarray_utils import uint_to_bitarray, bitarray_to_uint
from compressors.advanced_audio_coding.utilities import compute_mse, load_wav_audio, write_wav_audio, get_spectrogram

# importing files that contain the functions for the block diagram
from compressors.advanced_audio_coding.aac_huffman_coding import aac_huffman_encode, aac_huffman_decode, encode_prob_dist, decode_prob_dist
from compressors.advanced_audio_coding.filterbank import forward_filterbank, reverse_filterbank
from compressors.advanced_audio_coding.psychoacoustic_model import calculateThresholdsOnBlock, expand
from compressors.advanced_audio_coding.quantization import quantizeSimple, inverseQuantizeSimple


def bit_stream_format_encode(wav_data, fs=44100, channels=1, compression=1):
    """
    Inputs:
        - sample rate -> 4-byte int (default 44100 Hz)
        - num of channels -> 2-byte int (default 1, mono)
            - 1 for mono, 2 for stereo, etc
        - wav data: the raw input wav file data

    Outputs:
        - the bitstream
            - sample rate (16 bits)
            - channels (1 bit)
            - Quantized data: (32 bits)
            - probability distribution (32 bits)
            - the noiselssly coded spectra (any number of bits til the end)

    """
    bitstream = uint_to_bitarray(fs, bit_width=16) # sample rate is assumed 44100 Hz

    ### NOTE: num channels (currently assumes the input wav file is mono (not stereo)
    bitstream += uint_to_bitarray(channels, bit_width=1)

    bitstream += uint_to_bitarray(len(wav_data), bit_width=32) # add length of original wav file
    # call the filterbank, return windowed vals and the window shapes (which are assumed to be LONG)
    frame_length = 2048
    filtered_data = forward_filterbank(wav_data, frame_length)

    # quantization step
    # Call the psychoacoustic model and calculate the thresholds
    _, scaling_not_exp, scaling = calculateThresholdsOnBlock(filtered_data, compression=compression)

    # after psychoacoustic, goes into scaling and quantization   
    quant_spec = quantizeSimple(filtered_data, scaling)

    quant_spec_flattened=quant_spec.flatten()
    data_all = np.concatenate([quant_spec_flattened,scaling_not_exp.flatten()])
    # encode point between the two arrays
    bitstream += uint_to_bitarray(len(quant_spec_flattened), bit_width=32)

    # then huffman coding, which this gets added to bitstream
    # since encoded data is at the end, it can be however long
    encoded_data, prob_dist = aac_huffman_encode(data_all)

    # encode the probability distribution 
    bitstream += encode_prob_dist(prob_dist)
    bitstream += encoded_data

    return bitstream


def bit_stream_format_decode(bitstream):
    """
    Inputs:
        - the bitstream
            - sample rate (16 bits)
            - channels (1 bit)
            - the filterbank control info (window seq (default 1 bit (at most 2 bits)) and window shape (1 bit))
            - threshold values (32 bits)
            - the noiselssly coded spectra (any number of bits til the end)

    Outputs:
        - the uncompressed audio data
    
    """

    halflen = 1024
    idx = 16
    fs = bitarray_to_uint(bitstream[:idx])
    channels = bitarray_to_uint(bitstream[idx:idx+1])
    idx += 1
    n_samples = bitarray_to_uint(bitstream[idx: idx + 32])
    idx += 32

    # recover the quantization values from the bitstream
    quantization_values_num = bitarray_to_uint(bitstream[idx: idx + 32])
    idx += 32
    prob_dist, num_bits_read = decode_prob_dist(bitstream[idx:])
    idx += num_bits_read

    # Huffman decode
    decoded_data, num_bits_consumed = aac_huffman_decode(bitstream[idx:], prob_dist)
    idx += num_bits_consumed

    # retrieve the quantized values and scalings
    quant_values = decoded_data.data_list[:quantization_values_num]
    scalings = decoded_data.data_list[quantization_values_num:]
    scalings = expand(np.resize(scalings, (len(scalings)//49, 49)))
    inverse_quant_data = np.resize(np.array(quant_values), (len(quant_values)//halflen, halflen))
    filterbank_coefficients = inverseQuantizeSimple(inverse_quant_data, scalings)

    # Get back the original data 
    frame_length = 2048
    audio_data = reverse_filterbank(filterbank_coefficients, frame_length, n_samples)
    
    # theoretically, should get the audio data back
    return audio_data, fs

def run_compression(audio_name, compression=1):
    """
    Tests our implementation of AAC with different compression values (1, 2, 3, etc.)
    Inputs:
        - compression (Default 1): Takes in a compression value for calculating the thresholds
    
    This test function writes the wav files, prints out the MSE and kbps; 
    and writes the wav files for us to listen and compare it to the original
    """

    # import audio file (16 bit wav file)
    audio_arr, audio_sr = load_wav_audio(audio_name)

    # Encode audio file
    en_data = bit_stream_format_encode(audio_arr, fs=audio_sr, channels=1,compression=compression)

    # Decode audio file and write it to a wav file so we can listen to it
    dec_data, fs = bit_stream_format_decode(en_data)
    write_wav_audio('compressed_huffman'+str(compression)+'.wav', dec_data, fs)

    print('MSE', compute_mse(audio_arr, dec_data))
    print('kbps', len(en_data)/1024/(len(audio_arr)/audio_sr))
    
    # generates the spectrogram as pngs 
    # with file name from title (audio file name and compression value)
    title = 'Original WAV file'
    get_spectrogram(title, audio_arr, audio_sr)

    title2 = 'Compressed ' + 'with compression of ' + str(compression)
    get_spectrogram(title2, dec_data, fs)
    return dec_data

def test_end_to_end():
    """
    Tests the compression decoding and encoding end to end with dummy data.
    """
    # Generate random uniform data
    audio_arr = np.random.uniform(-0.5, 0.5, size=10000)
    sr = 44100

    # Encode audio file
    en_data = bit_stream_format_encode(audio_arr, fs=sr, channels=1,compression=1)

    # Decode audio file
    dec_data, dec_sr = bit_stream_format_decode(en_data)

    # Compute mse 
    mse = compute_mse(audio_arr, dec_data)
    # Check that mse error is reasonable
    assert mse < 1e-3
    
    # Check that we also decoded the sample rate correctly
    assert sr == dec_sr

def mpeg_spectrogram(mpeg_path):
    """
    Given a MPEG compressed AAC file, compare with our model by getting a spectrogram
    
    """
    mpeg_arr, mpeg_sr = load_wav_audio(mpeg_path)

    title = 'MPEG AAC'
    get_spectrogram(title, mpeg_arr, mpeg_sr)   


if __name__ == "__main__":
    # testing different quantization levels
    # test_end_to_end()
    audio_file = 'snippet.wav'
    for i in [1, 3, 7, 10, 20]:
        run_compression(audio_file, i)

    mpeg_file = 'ffmpeg_aac71k.wav'
    mpeg_spectrogram(mpeg_file)

