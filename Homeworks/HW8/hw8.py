import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv 
from numpy.linalg import norm

def make_nodes(N):
    a = []
    for i in range(N+1):
        a.append(5*np.cos(np.pi * (2*i+1) / (2*N)))
    a.reverse()
    print(a)
    return np.array(a)

def eval_lagrange(xeval,xint,yint,N):

    lj = np.ones(N+1)
    
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0.
    
    for jj in range(N+1):
       yeval = yeval + yint[jj]*lj[jj]
  
    return(yeval)
  
def lagrange():


    f = lambda x: 1 / (1 + x**2)

    N = 16
    ''' interval'''
    a = -5
    b = 5
   
   
    ''' create equispaced interpolation nodes'''
    # xint = np.linspace(a,b, N+1)
    xint = make_nodes(N)
    print(xint)
    ''' create interpolation data'''
    yint = f(xint)
    
    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval_l= np.zeros(Neval+1)
    yeval_dd = np.zeros(Neval+1)
  
    '''Initialize and populate the first columns of the 
     divided difference matrix. We will pass the x vector'''
    y = np.zeros( (N+1, N+1) )
     
    for j in range(N+1):
       y[j][0]  = yint[j]

    # y = dividedDiffTable(xint, y, N+1)
    ''' evaluate lagrange poly '''
    for kk in range(Neval+1):
       yeval_dd[kk] = eval_lagrange(xeval[kk],xint,yint,N)
    #    yeval_dd[kk] = bary(xeval[kk],xint,yint,N)
    #    yeval_dd[kk] = evalDDpoly(xeval[kk],xint,y,N)
          

    


    ''' create vector with exact values'''
    fex = f(xeval)
       
   
    plt.figure()    
    plt.title("Evals")
    plt.plot(xeval,fex,'ro-', label='f(x)')
    # plt.plot(xeval,yeval_l,'bs--') 
    plt.plot(xeval,yeval_dd,'c.--', label="Lagrange")
    plt.legend()

    plt.figure() 
    plt.title("Error")
    err_l = abs(yeval_l-fex)
    err_dd = abs(yeval_dd-fex)
    # plt.semilogy(xeval,err_l,'ro--',label='lagrange')
    plt.semilogy(xeval,err_dd,'bs--')
    plt.legend()
    plt.show()

def cubic_natural():
    f = lambda x: 1 / (1 + x**2)
    a, b = -5, 5
    Nint = 20
    xint = make_nodes(Nint) #np.linspace(a, b, Nint + 1)
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

def cubic_clamped():
    f = lambda x: 1 / (1 + x**2)
    a, b = -5, 5
    
    Nint = 16
    # xint = np.linspace(a, b, Nint + 1)
    xint = make_nodes(Nint)
    print(xint)
    yint = f(xint)

    Neval = 100
    xeval = np.linspace(xint[0], xint[Nint], Neval + 1)

    (M, C, D) = create_clamped_spline(yint, xint, Nint)
    yeval = eval_cubic_spline(xeval, Neval, xint, Nint, M, C, D)

    fex = f(xeval)
    nerr = norm(fex - yeval)
    print('nerr =', nerr)

    plt.figure()    
    plt.plot(xeval, fex, 'ro-', label='exact function')
    plt.plot(xeval, yeval, 'bs--', label='clamped spline') 
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

def create_clamped_spline(yint, xint, N):
    b = np.zeros(N + 1)
    h = np.diff(xint)

    for i in range(1, N):
        b[i] = (yint[i + 1] - yint[i]) / h[i] - (yint[i] - yint[i - 1]) / h[i - 1]

    A = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        if i == 0:
            A[i, 0] = h[0] / 3
            A[i, 1] = h[0] / 6
        elif i == N:
            A[i, N] = h[i-1] / 3
            A[i, N-1] = h[i-1] / 6
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

cubic_natural()
# cubic_clamped()
# lagrange()
# hermite()
