import numpy as np
from scalefactor_bands import get_scalefactor_bands

def calculateThresholds(X, s_h, s_l, thr_q_prev,thr_quiet):
    """
    Calculate psychoacoustic threshold thr(n), an upper limit for the
    quantization noise of the coder.

    Inputs:
    X: filterbank output
    scalefactor: list of scalefactor indicies
    s_h and s_l: output from spreaded energy calculation
    thr_q_prev: the thr_q value from the previous timestep
    thr_quiet: thresholds in quiet

    Outputs:
    thr: threshold vlaues for each scalefactor band
    thr_q: used for next calculateThresholds calculation

    """
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
    """
    for n in range(N):
        thr_q[n] = max(thr_spr[n],thr_quiet[n])
    """
    thr_q = thr_spr
    # pre-echo control
    rpelev = 2
    rpmin = 0.01
    thr = np.zeros(N)
    for n in range(N):
        thr[n] = max(rpmin*thr_q[n],min(thr_q[n],rpelev*thr_q_prev) )
    
    return thr, thr_q


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
    s_l, s_h = calculateSpreadedEnergy()
    print(calculateThresholds(1000*np.ones([1024]),s_h,s_l,0,0))