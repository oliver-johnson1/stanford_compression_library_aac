
import numpy as np
import tensorflow as tf

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
    frame_length = 2048
    frame_step =  frame_length// 2

    # Calculate number of frames, using double negatives to round up.
    num_frames = -(-N // frame_step)
    print(num_frames,'num_frames')

    # Pad the inner dimension of raw_data by pad_samples.
    pad_samples = max(0, frame_length + frame_step * (num_frames - 1) - N)
    paddings = [[0, pad_samples]]
    print(paddings,'tf paddings shape')
    raw_data = tf.pad(raw_data, paddings)

    raw_data_shape = tf.shape(raw_data)
    print(raw_data_shape,'tf raw_data shape')

    subframe_length = 1024
    subframes_per_frame = frame_length // subframe_length
    subframes_per_hop = frame_step // subframe_length
    num_subframes = raw_data_shape[0] // subframe_length

    slice_shape = raw_data_shape
    subframe_shape = [num_subframes, subframe_length]
    print(subframe_shape,'subframe shape')
    subframes = tf.reshape(tf.strided_slice(
        raw_data, tf.zeros_like(raw_data_shape),
        slice_shape), subframe_shape)

    # frame_selector is a [num_frames, subframes_per_frame] tensor
    frame_selector = tf.reshape(
        np.arange(num_frames) *
        subframes_per_hop, [num_frames, 1])
    print(frame_selector.shape,'frame select shape tf')

    # subframe_selector is a [num_frames, subframes_per_frame] tensor
    subframe_selector = tf.reshape(
        np.arange(subframes_per_frame),
        [1, subframes_per_frame])
    print(subframe_selector.shape, 'subframe selectro shape tf')

    selector = frame_selector + subframe_selector

    print(subframes.shape,'subframe shape',selector.shape)
    frames = tf.reshape(
        tf.gather(subframes, selector, axis=0),
        [num_frames, frame_length],0)
    print(frames.shape,'ending frames shape')
    frames = frames.numpy()

    return frames
