### NOTE: acts as an intermediary to use 
# Stanford Compression Library's Huffman

from core.prob_dist import ProbabilityDist
from compressors.huffman_coder import HuffmanEncoder, HuffmanDecoder

class Histogram:
    def __init__(self):
        self.LOW_FREQ = 10
        self.ESCAPE_CODE = -1
        # Stores the occurence probability of each symbol
        self.probability = dict()
        # Stores the occurence frequency of each symbol
        self.statistics = dict()        
    