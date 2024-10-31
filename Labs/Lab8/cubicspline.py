import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv 
from numpy.linalg import norm

def driver():
    f = lambda x: 1 / (1 + 100*x**2)
    a, b = -1, 1
    Nint = 15
    xint = np.linspace(a, b, Nint + 1)
    yint = f(xint)

    Neval = 100
    xeval = np.linspace(xint[0], xint[Nint], Neval + 1)

    (M, C, D) = create_natural_spline(yint, xint, Nint)
    yeval = eval_cubic_spline(xeval, Neval, xint, Nint, M, C, D)

    fex = f(xeval)
    nerr = norm(fex - yeval)
    print('nerr =', nerr)

    plt.figure()    
    plt.plot(xeval, fex, 'ro-', label='exact function')
    plt.plot(xeval, yeval, 'bs--', label='natural spline') 
    plt.legend()
    plt.show()

    err = abs(yeval - fex)
    plt.figure() 
    plt.semilogy(xeval, err, 'ro--', label='absolute error')
    plt.legend()
    plt.show()

def create_natural_spline(yint, xint, N):
    b = np.zeros(N + 1)
    h = np.diff(xint)

    for i in range(1, N):
        b[i] = (yint[i + 1] - yint[i]) / h[i] - (yint[i] - yint[i - 1]) / h[i - 1]

    A = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        if i == 0:
            A[i, 0] = 1
        elif i == N:
            A[i, N] = 1
        else:
            A[i, i - 1] = h[i - 1] / 6
            A[i, i] = (h[i] + h[i - 1]) / 3
            A[i, i + 1] = h[i] / 6

    M = np.linalg.solve(A, b)

    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
        C[j] = yint[j] / h[j] - h[j] * M[j] / 6
        D[j] = yint[j + 1] / h[j] - h[j] * M[j + 1] / 6
    return M, C, D

def eval_local_spline(xeval, xi, xip, Mi, Mip, C, D):
    hi = xip - xi
    yeval = ((xip - xeval)**3 * Mi) / (6 * hi) + (xeval - xi)**3 * (Mip / (6 * hi)) + C * (xip - xeval) + D * (xeval - xi)
    return yeval 

def eval_cubic_spline(xeval, Neval, xint, Nint, M, C, D):
    yeval = np.zeros(Neval + 1)
    
    for j in range(Nint):
        atmp = xint[j]
        btmp = xint[j + 1]
        
        ind = np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]

        yloc = eval_local_spline(xloc, atmp, btmp, M[j], M[j + 1], C[j], D[j])
        yeval[ind] = yloc

    return yeval

driver()
