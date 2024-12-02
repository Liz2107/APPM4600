import numpy as np

def powerIt(A, Nmax):
    n = A.shape[0]
    q = np.random.rand(n)
    q = q / np.linalg.norm(q)
    lam = np.zeros(Nmax)
    for j in range(Nmax):
        z = A @ q
        q = z / np.linalg.norm(z)
        lam[j] = q.T @ A @ q
    v = q
    return lam, v

import scipy.linalg

H = scipy.linalg.hilbert(16)
# H = np.array([[1,1],[0,1]])
lam, v = powerIt(np.linalg.inv(H), 20)
# print(1/lam)
print(1/lam, v)
error = np.linalg.norm(H*v - (-6.10326534e-19)*v)
print(error)