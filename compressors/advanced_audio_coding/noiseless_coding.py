### NOTE, Huffman coding is used to represent 
# n-tuples of quantized coeffs
# with Huffman code drawn from one of 11 codebooks
## the magnitude of the coeffs is huffman coded

# from compressors.huffman_coder import HuffmanEncoder, HuffmanDecoder

def noiseless_encoding(spec_coeffs):
    """
    Input:
        - spec_coeffs: set of 1024 quantized spectral coefficients
    
    Output:
        - 

    Steps:
        - Spectrum clipping
        - Preliminary huffman coding using max number of sections
        - Section merging to achieve lowest bit count
    """
    # Spectrum clipping

    ### NOTE: Up to four coefficients can be 
    # coded separately as magnitudes in excess of one
    ## a value of +-1 left in the quantized
    # coefficient array to carry the sign
    ## the index of the scalefactor band 
    # containing the lowest-frequency
    ##  Each of the “clipped” coefficients is coded as 
    # a magnitude (in excess of 1: clip those freq coeffs' magnitudes greater than 1)
    # and an offset from the base of the previously indicated 
    # scalefactor band.

    # Sectioning



def noiseless_decoding():
    """
    Inputs:
        - Huffman coded 4-tuples or 2-tupes of quantized spectral coeff

            unsigned = Boolean value unsigned_cb[i], listed in second column of Table 59.
            dim = Dimension of codebook, listed in the third column of Table 59.
            lav = Largest abs val able to be encdoed by ea 
                    codebk and defines bool helper var arr 
                    (unsigned_cb: 0 (signed) or 1 (unsigned)), 
                    listed in the fourth column of Table 59.
            idx = codeword index

    Outputs:
        - 

    """

    if unsigned:
        mod_lav = lav + 1
        off = 0
    else:
        mod_lav = 2*lav + 1
    
    if dim == 4:
        w = int(idx/(mod_lav**3)) - off
        idx -= (w + off) * (mod_lav**3)
        x = int(idx/(mod_lav**2)) - off
        idx -= (x + off) * (mod_lav**2)
        y = int(idx/(mod_lav)) - off
        idx -= (y + off) * (mod_lav)
        z = idx - off
    else:
        y = int(idx/mod_lav) - off
        idx -= (y + off) * (mod_lav)
        z = idx - off