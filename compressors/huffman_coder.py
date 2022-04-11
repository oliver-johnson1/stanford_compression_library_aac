from dataclasses import dataclass
from typing import Any, Tuple
import heapq
from functools import total_ordering
from compressors.prefix_free_compressors import PrefixFreeTree, PrefixFreeEncoder, PrefixFreeDecoder
from core.prob_dist import ProbabilityDist
import numpy as np
from utils.bitarray_utils import BitArray
from utils.test_utils import get_random_data_block, try_lossless_compression
from utils.tree_utils import BinaryNode


@dataclass
@total_ordering  # decorator which adds other compare ops give one
class HuffmanNode(BinaryNode):
    """represents a node of the huffman tree

    NOTE: BinaryNode class already has left_child, right_child, id fields
    here by subclassing we add the field: prob
    """

    prob: float = None

    def __le__(self, other):
        """
        Define a comparison operator, so that we can use this while comparing nodes
        # NOTE: we only need to define one compare op, as others can be implemented using the
        decorator @total_ordering
        """
        return self.prob <= other.prob


class HuffmanTree(PrefixFreeTree):
    def __init__(self, prob_dist: ProbabilityDist):
        self.prob_dist = prob_dist

        # construct the tree and set the root_node of PrefixFreeTree base class
        super().__init__(root_node=self.build_huffman_tree())

    def build_huffman_tree(self) -> HuffmanNode:
        """Build the huffman coding tree

        1. Sort the prob distribution, combine last two symbols into a single symbol
        2. Continue until a single symbol is left
        """
        # Lets say we have symbols {1,2,3,4,5,6} with prob {p1, p2,...p6}
        # We first start by initializing a list
        # [..., HuffmanNode(id=3, prob=p3), (HuffmanNode(id=6, prob=p6),  ...]

        node_list = []
        for a in self.prob_dist.alphabet:
            node = HuffmanNode(id=a, prob=self.prob_dist.probability(a))
            node_list.append(node)

        # create a node_heap from the node_list (in place)
        # NOTE: We create a min-heap data structure to represent the list, as
        # We are concerned about finding the top two smallest elements from the list
        # Heaps are efficient at such operations O(log(n)) -> push/pop, O(1) -> min val
        node_heap = node_list  # shallow copy
        heapq.heapify(node_heap)

        while len(node_heap) > 1:
            # create a min-heap (in-place) from the node list, so that we can get the
            heapq.heapify(node_heap)

            # get the two smallest symbols
            last1 = heapq.heappop(node_heap)
            last2 = heapq.heappop(node_heap)

            # insert a symbol with the sum of the two probs
            combined_prob = last1.prob + last2.prob
            combined_node = HuffmanNode(left_child=last1, right_child=last2, prob=combined_prob)
            heapq.heappush(node_heap, combined_node)

        # finally the node_prob_dist should contain a single element
        assert len(node_heap) == 1

        # return the huffman tree
        # only one element should remain
        root_node = node_heap[0]
        return root_node


class HuffmanEncoder(PrefixFreeEncoder):
    '''
    PrefiFreeEncoder already has a encode_block function to encode the symbols once we define a encode_symbol function
    for the particular compressor.
    '''
    def __init__(self, prob_dist: ProbabilityDist):
        self.tree = HuffmanTree(prob_dist)
        # self.encoding_table = tree.get_encoding_table()

    def encode_symbol(self, s):
        return self.tree.encode_symbol(s)


class HuffmanDecoder(PrefixFreeDecoder):
    def __init__(self, prob_dist: ProbabilityDist):
        self.tree = HuffmanTree(prob_dist)

    def decode_symbol(self, encoded_bitarray: BitArray) -> Tuple[Any, BitArray]:
        decoded_symbol, num_bits_consumed = self.tree.decode_symbol(encoded_bitarray)
        return decoded_symbol, num_bits_consumed


def test_huffman_coding_dyadic():
    """test huffman coding on dyadic distributions

    On dyadic distributions Huffman coding should be perfectly equal to entropy
    1. Randomly generate data with the given distribution
    2. Construct Huffman coder using the given distribution
    3. Encode/Decode the block
    """
    NUM_SAMPLES = 10000

    distributions = [
        ProbabilityDist({"A": 0.5, "B": 0.5}),
        ProbabilityDist({"A": 0.5, "B": 0.25, "C": 0.25}),
        ProbabilityDist({"A": 0.5, "B": 0.25, "C": 0.125, "D": 0.125}),
    ]
    print()
    for prob_dist in distributions:
        # generate random data
        data_block = get_random_data_block(prob_dist, NUM_SAMPLES, seed=0)

        # create encoder decoder
        encoder = HuffmanEncoder(prob_dist)
        decoder = HuffmanDecoder(prob_dist)

        # perform compression
        is_lossless, output_len, _ = try_lossless_compression(data_block, encoder, decoder)
        avg_bits = output_len / NUM_SAMPLES

        assert is_lossless, "Lossless compression failed"
        np.testing.assert_almost_equal(
            avg_bits,
            prob_dist.entropy,
            decimal=2,
            err_msg="Huffman coding is not close to entropy",
        )
        print(f"Avg Bits: {avg_bits}, Entropy: {prob_dist.entropy}")
