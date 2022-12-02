### NOTE: acts as an intermediary to use 
# Stanford Compression Library's Huffman (pg. 186)

from core.prob_dist import ProbabilityDist
from compressors.huffman_coder import HuffmanEncoder, HuffmanDecoder
from utils.bitarray_utils import BitArray, uint_to_bitarray


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

    encoding = BitArray()
    for s in quant_spec:
        encoding += huff_en.encode_symbol(s) #assuming encode sym returns a bitarray rn

    return encoding

def aac_huffman_decode(encoded_bitarray, prob_dist):
    huff_dec = HuffmanDecoder(prob_dist)
    quant_spec = []
    for b in encoded_bitarray:
        quant_spec.append(huff_dec.decode_symbol(b))
    return quant_spec