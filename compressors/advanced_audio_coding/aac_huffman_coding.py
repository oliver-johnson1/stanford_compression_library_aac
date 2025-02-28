# Acts as an intermediary to feed into 
# Stanford Compression Library's Huffman Class

from core.prob_dist import ProbabilityDist
from compressors.huffman_coder import HuffmanEncoder, HuffmanDecoder
from utils.bitarray_utils import BitArray, uint_to_bitarray, bitarray_to_uint
import pickle
from core.data_block import DataBlock
import numpy as np
   
def generate_prob_dist(quant_spec):
    """
    Generating a probability dist of the quantized spectra to generate the codebooks for Huffman coding

    Inputs:
        - quant_spect: The quantized spectra data
    Outputs:
        - Probability distribution
    """
    unique, counts = np.unique(quant_spec, return_counts=True)
    total_count = sum(counts)
    prob_dict = dict(zip(unique, counts/total_count))    
    # Convert dict to ProbabilityDist
    prob_dist = ProbabilityDist({k: prob_dict[k] for k in prob_dict})
    return prob_dist

def aac_huffman_encode(quant_spec):
    """
    Huffman encodes the quantized spectra using the Stanford Compression Library's class
    """
    prob_dist = generate_prob_dist(quant_spec)

    huff_en = HuffmanEncoder(prob_dist)

    encoded_bitarray = huff_en.encode_block(DataBlock(list(quant_spec)))

    return encoded_bitarray, prob_dist

def aac_huffman_decode(encoded_bitarray, prob_dist):
    """
    Huffman decodes the encoded bitarray using the Stanford Compression Library's class
    to get back the quantized spectra
    """
    huff_dec = HuffmanDecoder(prob_dist)
    decoded_block, num_bits_consumed = huff_dec.decode_block(encoded_bitarray)
    return decoded_block, num_bits_consumed

def encode_prob_dist(prob_dist: ProbabilityDist) -> BitArray:
    """Encode a probability distribution as a bit array

    Args:
        prob_dist (ProbabilityDist): probability distribution over 0, 1, 2, ..., 255
            (note that some probabilities might be missing if they are 0).

    Returns:
        BitArray: encoded bit array
    """
    # pickle prob dist and convert to bytes
    pickled_bits = BitArray()
    pickled_bits.frombytes(pickle.dumps(prob_dist))
    len_pickled = len(pickled_bits)
    # encode length of pickle
    length_bitwidth = 32
    length_encoding = uint_to_bitarray(len_pickled, bit_width=length_bitwidth)

    encoded_probdist_bitarray = length_encoding + pickled_bits

    return encoded_probdist_bitarray

def decode_prob_dist(bitarray: BitArray):
    """Decode a probability distribution from a bit array

    Args:
        bitarray (BitArray): bitarray encoding probability dist followed by arbitrary data

    Returns:
        prob_dit (ProbabilityDist): the decoded probability distribution
        num_bits_read (int): the number of bits read from bitarray to decode probability distribution
    """
    # first read 32 bits from start to get the length of the pickled sequence
    length_bitwidth = 32
    length_encoding = bitarray[:length_bitwidth]
    len_pickled = bitarray_to_uint(length_encoding)
    # bits to bytes
    pickled_bytes = bitarray[length_bitwidth: length_bitwidth + len_pickled].tobytes()

    prob_dist = pickle.loads(pickled_bytes)
    num_bits_read = length_bitwidth + len_pickled
    return prob_dist, num_bits_read