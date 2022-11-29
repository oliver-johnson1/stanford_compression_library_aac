import numpy as np


# Calculation of the energy spectrum
en = np.zeros(N)
for n in N:
    X_n = X[scalefactor[n]:scalefactor[n+1]]
    en[n] = np.sum(X_n*X_n)


# from energy to threshold
SNR = 29 #dB
thr_scaled = en/SNR

# spreading
thr_spr_prime = np.zeros(N)
thr_spr = np.zeros(N)
for n in N:
    thr_spr_prime[n] = max(thr_scaled[n],s_k[n]*thr_scaled[n-1])

for n in N:
    thr_spr[n] = max(thr_spr_prime[n],s_l[n]*thr_spr_prime[n+1])

# threshold in quiet
thr_q_prev = thr_q
for n in N:
    thr_q[n] = max(thr_spr[n],thr_quiet[n])

# pre-echo control
rpelev = 2
rpmin = 0.01
for n in N:
    thr[n] = max(rpmin*thr_q[n],min(thr_q[n],rpelev*thr_q_prev) )