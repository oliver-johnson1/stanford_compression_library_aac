
import numpy as np
import tensorflow as tf
def forward_filterbank(data, frame_length=2048):
    """
    Computes the forward MDCT on the data

    Inputs:
        - data: the raw wav file array
        - frame_length (default 2048): The window length
    Outputs:
        - the new filtered data
    """
    # pads the data by surrounding it with zeros 
    waveform_pad = tf.pad(data.astype(float), [[frame_length//2, 0],])
    # feed this padded data into the mdct using the KBD window
    filtered_data = np.array(tf.signal.mdct(waveform_pad, frame_length, pad_end=True,
                            window_fn=tf.signal.kaiser_bessel_derived_window))
    return filtered_data

def reverse_filterbank(filtered_data, frame_length, n_samples):
    """
    Computes the inverse MDCT on the data

    Inputs:
        - filtered_data: the quantized spectra data
        - frame_length (default 2048): The window length
    Outputs:
        - the recovered data
    """
    # takes the inverse mdct with the KBD window
    inverse_mdct = np.array(tf.signal.inverse_mdct(filtered_data,
                                            window_fn=tf.signal.kaiser_bessel_derived_window))
    # unpads the data, to get back the original signal                                            
    recovered_data = inverse_mdct[frame_length//2:frame_length//2+n_samples]
    return recovered_data

def test_filterbank():
    """
    Tests to make sure what we get back what weput into the filterbank forward and reverse, 
    """

    N = 10000
    data = np.random.standard_normal(size=N)
    frame_length = 2048
    filtered_data = forward_filterbank(data,frame_length)
    recovered_data = reverse_filterbank(filtered_data, frame_length, N)
    assert np.allclose(data,recovered_data,rtol=1e-8,atol=1e-8)


if __name__ == "__main__":
    test_filterbank()