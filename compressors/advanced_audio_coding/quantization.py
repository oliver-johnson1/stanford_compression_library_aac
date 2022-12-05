import numpy as np
MAGIC_NUMBER = 0.4054
globalGain = 0
# Scalefactor determination



def forwardQuantization(X, scf):
    N = len(X)
    X_quant = np.zeros(N)
    for k in range(N):
        X_quant[k] = np.sign(X[k])*int((np.abs(X[k])*2**(0.25*(scf[k]-globalGain)))**0.75 + MAGIC_NUMBER)
    #print(X_quant)
    return X_quant

def inverseQuantization(X_quant, scf):
    N = len(X_quant)
    X_invquant = np.zeros(N)
    for k in range(N):
        X_invquant[k] = np.sign(X_quant[k])*np.abs(X_quant[k])**(4/3)*2**(-0.25*(scf[k]-globalGain))
    return X_invquant

def forwardQuantizationBlock(B,scf_b):
    quantizedBlock = []
    for X, scf in zip(B, scf_b):
        quantizedBlock.append(forwardQuantization(X,scf))
    return quantizedBlock

def inverseQuantizationBlock(qB, scf_b):
    B = []
    for q, scf in zip(qB, scf_b):
        i = inverseQuantization(q,scf)
        B.append(i)
    return B
# Quantization functions from the assignment
def quantize(A, nlevels):
    """
    Quantize array A to nlevels.
    
    Returns the quantized values and the quantization limits.
    """
    smallest = np.min(A)
    largest = np.max(A)
    bins = np.linspace(smallest, largest, nlevels+1)
    bins[-1] = np.nextafter(bins[-1], np.infty)
    return np.digitize(A, bins), (smallest, largest)

def unquantize(Q, nlevels, limits):
    """
    Decoder of quantize. Takes as input the quantized output, the
    number of levels and the quantization limits.
    
    Returns the unquantized array.
    """
    smallest, largest = limits
    bins = np.linspace(smallest, largest, nlevels+1)
    vals = (bins[1:]+bins[:-1]) / 2
    return vals[Q-1]

def forwardQuantization2(X, scalefactors):
    MAX_QUANT = 8191

    start_common_scalefac = np.ceil(16/3*np.log2((max_mdct_line**0.75)/MAX_QUANT))
    scalefactor = np.zeros(10)
    mean_bits = bit_ratee * 1024 / sampling_rate
    if more_bits > 0 :
        available_bits = mean_bits + min ( more_bits, bit_reservoir_state[frame])
    if more_bits < 0 :
        available_bits = mean_bits + max ( more_bits, bit_reservoir_state[frame]- max_bit_reservoir)

if __name__ == "__main__":
    X = [np.random.normal(size=1024)*1000,np.random.normal(size=1024)*1500]
    scf = [np.ones(1024)*20,np.ones(1024)*20]
    X_quant = forwardQuantizationBlock(X, scf)
    print(X_quant)
    X_invquant = inverseQuantizationBlock(X_quant, scf)
