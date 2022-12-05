### NOTE: acts as an intermediary to use 
# Stanford Compression Library's Huffman (pg. 186)

from core.prob_dist import ProbabilityDist
from compressors.huffman_coder import HuffmanEncoder, HuffmanDecoder
from utils.bitarray_utils import BitArray, uint_to_bitarray, bitarray_to_uint
import pickle
import numpy as np
from compressors.universal_uint_coder import UniversalUintEncoder, UniversalUintDecoder
from core.data_block import DataBlock


# class Histogram:
#     def __init__(self):
#         self.LOW_FREQ = 10
#         self.ESCAPE_CODE = -1
#         # Stores the occurence probability of each symbol
#         self.probability = dict()
#         # Stores the occurence frequency of each symbol
#         self.statistics = dict()        

#         # Convert dict to ProbabilityDist
#         prob_dist = ProbabilityDist({k: prob_dict[k] for k in prob_dict})
    
def generate_prob_dist(quant_spec):
    """
    Generating a probability dist of the quantized spectra to generate the codebooks for Huffman coding
    """
    stats = {}
    prob_dict = {}
    for i in range(len(quant_spec)):
        if quant_spec[i] in stats:
            stats[quant_spec[i]] = stats[quant_spec[i]] + 1
        else:
            stats[quant_spec[i]] = 1
    total_count = sum(stats.values())
    for key, val in stats.items():
        prob_dict[key] = val/float(total_count)
    
    # Convert dict to ProbabilityDist
    prob_dist = ProbabilityDist({k: prob_dict[k] for k in prob_dict})
    return prob_dist

def aac_huffman_encode(quant_spec):
    prob_dist = generate_prob_dist(quant_spec)

    huff_en = HuffmanEncoder(prob_dist)

    # encoding = BitArray()
    encoded_bitarray = huff_en.encode_block(DataBlock(list(quant_spec)))
    # for s in quant_spec:
    #     encoding += huff_en.encode_symbol(s) #assuming encode sym returns a bitarray rn

    return encoded_bitarray, prob_dist

def aac_huffman_decode(encoded_bitarray, prob_dist):
    huff_dec = HuffmanDecoder(prob_dist)
    decoded_block, num_bits_consumed = huff_dec.decode_block(encoded_bitarray)
    # quant_spec = []
    
    # for b in encoded_bitarray:
    #     quant_spec.append(huff_dec.decode_symbol(b))
    return decoded_block, num_bits_consumed

def encode_prob_dist(prob_dist: ProbabilityDist) -> BitArray:
    """Encode a probability distribution as a bit array

    Args:
        prob_dist (ProbabilityDist): probability distribution over 0, 1, 2, ..., 255
            (note that some probabilities might be missing if they are 0).

    Returns:
        BitArray: encoded bit array
    """
    #########################
    # ADD CODE HERE
    # bits = BitArray(), bits.frombytes(byte_array), uint_to_bitarray might be useful to implement this
    # raise NotImplementedError("You need to implement encode_prob_dist")

    # pickle prob dist and convert to bytes
    pickled_bits = BitArray()
    pickled_bits.frombytes(pickle.dumps(prob_dist))
    len_pickled = len(pickled_bits)
    # encode length of pickle
    length_bitwidth = 32
    length_encoding = uint_to_bitarray(len_pickled, bit_width=length_bitwidth)

    encoded_probdist_bitarray = length_encoding + pickled_bits
    #########################

    return encoded_probdist_bitarray

def decode_prob_dist(bitarray: BitArray):
    """Decode a probability distribution from a bit array

    Args:
        bitarray (BitArray): bitarray encoding probability dist followed by arbitrary data

    Returns:
        prob_dit (ProbabilityDist): the decoded probability distribution
        num_bits_read (int): the number of bits read from bitarray to decode probability distribution
    """
    #########################
    # ADD CODE HERE
    # bitarray.tobytes() and bitarray_to_uint() will be useful to implement this
    # raise NotImplementedError("You need to implement decode_prob_dist")
    # first read 32 bits from start to get the length of the pickled sequence
    length_bitwidth = 32
    # print(bitarray)
    length_encoding = bitarray[:length_bitwidth]
    len_pickled = bitarray_to_uint(length_encoding)
    # bits to bytes
    pickled_bytes = bitarray[length_bitwidth: length_bitwidth + len_pickled].tobytes()
    # print('pickled_bytes', pickled_bytes)
    prob_dist = pickle.loads(pickled_bytes)
    num_bits_read = length_bitwidth + len_pickled
    #########################
    return prob_dist, num_bits_read