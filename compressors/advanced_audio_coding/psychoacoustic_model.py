import numpy as np
from compressors.advanced_audio_coding.scalefactor_bands import get_scalefactor_bands, get_theshold_in_quiet

def calculateThresholdsOnBlock(B, compression = 1):

    """
    Computes the thresholds on the whole filterbank output
    Inputs:
    B: block of filterbank outputs
    compression: parameter to decide the level of quantization higher means that
                 there will be more compression and more quantization noise
    
    Outputs:
    thresholds: raw thresholds values
    thresholds_scaled: thresholds scaled by log factor
    thesholds_scaled_expanded: thresholds expanded to cover entire freq range
    """

    thresholds = []
    for X in B:
        thr = calculateThresholds(X)
        thresholds.append(thr)
        
    thresholds = np.array(thresholds)

    # scale thesholds with square root function so that there is a less significant 
    # difference between large and small values
    thesholds_scaled = np.round(compression*np.sqrt(thresholds))+1

    # as thesholds are only defined for scale factors (a range of frequencies), we 
    # expand this so a value is available for each frequency

    thesholds_scaled_expanded = expand(thesholds_scaled)

    return thresholds, thesholds_scaled, thesholds_scaled_expanded

def expand(scaling):
    """
    Expands scale factors to whole frequency range
    Input: 
    scaling: scalefactor coefficients

    Output:
    scalefactor coefficents but expanded to each frequency
    """

    scaling_expanded = np.zeros((scaling.shape[0], 1024))
    scalefactor, N = get_scalefactor_bands()
    for n in range(N):
        scaling_expanded[:,scalefactor[n]:scalefactor[n+1]]=np.expand_dims(scaling[:,n],axis=1)
    return scaling_expanded


def calculateThresholds(X):
    """ 
    Calculate psychoacoustic threshold thr(n), an upper limit for the
    quantization noise of the coder.

    Inputs:
    X: filterbank output

    Outputs:
    thr: threshold vlaues for each scalefactor band

    """
    # Calculate spreaded energy
    s_l, s_h = calculateSpreadedEnergy()

    # Calculation of the energy spectrum
    scalefactor, N = get_scalefactor_bands()
    en = np.zeros(N)
    for n in range(N):
        X_n = X[scalefactor[n]:scalefactor[n+1]]
        en[n] = np.sum(X_n*X_n)


    # from energy to threshold
    SNR = 29 #dB
    SNR_linear = 10**(SNR/10)
    thr_scaled = en/SNR_linear

    # spreading
    thr_spr_prime = np.zeros(N)
    
    for n in range(N):
        if n == 0:
            thr_spr_prime[n] = thr_scaled[n]
        else:
            thr_spr_prime[n] = max(thr_scaled[n],s_h[n]*thr_scaled[n-1])
    thr_spr = np.zeros(N)
    for n in range(N):
        if n == N-1:
            thr_spr[n] = thr_spr_prime[n]
        else:
            thr_spr[n] = max(thr_spr_prime[n],s_l[n]*thr_spr_prime[n+1])

    # threshold in quiet
    thr_q = np.zeros(N)
    thr_quiet = get_theshold_in_quiet()
    for n in range(N):
        thr_q[n] = max(thr_spr[n], thr_quiet[n])
    
    rpmin = 0.01
    thr = np.zeros(N)
    for n in range(N):
        thr[n] = rpmin*thr_q[n]
    # high threshold -> more quantization
    return thr


def calculateSpreadedEnergy(bitrate=44100):
    """
    Spreaded Energy Calculation as per section 5.4.3
    Only consider long blocks
    """

    scalefactor, N = get_scalefactor_bands()
    scalefactor_bark = freq_to_bark(scalefactor)
    delta_bark = np.diff(scalefactor_bark)
    #
    s_l = 30 * delta_bark
    if bitrate > 22000:
        s_h = 100 * delta_bark # or 10**(20/10)=100
    else:
        s_h = 31.62 * delta_bark # or 10**(15/10)=31.62
    return s_l, s_h

def freq_to_bark(freq):
    return 13 * np.arctan(0.00076 * freq) + 3.5 * np.arctan((freq/7500)**2)


if __name__ == "__main__":
    print(calculateThresholds(1000*np.ones([1024])))