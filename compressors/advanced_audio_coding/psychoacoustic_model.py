import numpy as np

def calculateThresholds(X, scalefactor, s_h, s_l, thr_q_prev,thr_quiet):
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
    N = len(scalefactor) - 1 ###
    en = np.zeros(N)
    for n in N:
        X_n = X[scalefactor[n]:scalefactor[n+1]]
        en[n] = np.sum(X_n*X_n)


    # from energy to threshold
    SNR = 29 #dB
    SNR_linear = 10**(SNR/10)
    thr_scaled = en/SNR_linear

    # spreading
    thr_spr_prime = np.zeros(N)
    
    for n in N:
        thr_spr_prime[n] = max(thr_scaled[n],s_h[n]*thr_scaled[n-1])
    thr_spr = np.zeros(N)
    for n in N:
        thr_spr[n] = max(thr_spr_prime[n],s_l[n]*thr_spr_prime[n+1])

    # threshold in quiet
    thr_q = np.zeros(N)
    for n in N:
        thr_q[n] = max(thr_spr[n],thr_quiet[n])

    # pre-echo control
    rpelev = 2
    rpmin = 0.01
    thr = np.zeros(N)
    for n in N:
        thr[n] = max(rpmin*thr_q[n],min(thr_q[n],rpelev*thr_q_prev) )
    
    return thr, thr_q