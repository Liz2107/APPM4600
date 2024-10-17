#Pre-lab write up

'''
The simple way to solve this problem is to construct a Vandemond matrix and solve the associated system when multiplied by a vector of the coefficients (a_0...a_n)
Summary: 
We will test the three methods we have learned with different nodes to see how they perform. We plot an initial test. 
We analyze the results as we increase N, paying attention to the error. 
We then pick the best method and redefine the nodes. We see how this affects the error. 
We summarize the results and push the code to gihub.
'''

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import norm

#Pre-lab code
def q1b(): 

    f = lambda x: 1 / (1 + (10*x)**2)
    
    N = 15
    a = -1
    b = 1
    
    xint = np.linspace(a,b,N+1)
    print('xint =',xint)
    yint = f(xint)
    print('yint =',yint)
    
    V = Vandermonde(xint,N)  
    Vinv = inv(V)
    
    ''' Apply inverse to rhs'''
    ''' to create the coefficients'''
    coef = Vinv @ yint

# No validate the code
    Neval = 1000    
    xeval = np.linspace(a,b,Neval+1)
    yeval = eval_monomial(xeval,coef,N,Neval)

# exact function
    yex = f(xeval)

    err =  norm(yex-yeval) 
    # print('err = ', err)
    
    plt.plot(xeval, yeval, label="Interp")
    plt.scatter(xint, f(xint), label= "Evaluated Data Points", color="green")
    plt.plot(xeval, yex, label="Actual Function")
    
    # plt.plot(xeval, np.abs(yex-yeval), label="Absolute Error")
    plt.legend()
    plt.show()
    return

def  eval_monomial(xeval,coef,N,Neval):

    yeval = coef[0]*np.ones(Neval+1)
    
#    print('yeval = ', yeval)
    
    for j in range(1,N+1):
      for i in range(Neval+1):
        yeval[i] = yeval[i] + coef[j]*xeval[i]**j

    return yeval

   
def Vandermonde(xint,N):
    V = np.zeros((N+1,N+1))
    
    for j in range(N+1):
       V[j][0] = 1.0

    for i in range(1,N+1):
        for j in range(N+1):
           V[j][i] = xint[j]**i

    return V     
# q1b()

def make_nodes(N):
    a = []
    for i in range(N+1):
        a.append(np.cos(np.pi * (2*i+1) / (2*N)))
    return np.array(a)

def driver():


    f = lambda x: 1 / (1 + (10*x)**2)

    N = 20
    ''' interval'''
    a = -1
    b = 1
   
   
    ''' create equispaced interpolation nodes'''
    # a = np.array([((2*j - 1) / 2*N) for j in range(1, N+1)])
    # print(np.cos(a*np.pi))
    # xint = np.array([1 + np.cos(np.pi * (2*j - 1)/2*N) for j in range(1,N+1)])
    # # print(xint)
    # print(a)
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
    plt.plot(xeval,yeval_dd,'c.--', label="Barycentric Lagrange")
    plt.legend()

    plt.figure() 
    plt.title("Error")
    err_l = abs(yeval_l-fex)
    err_dd = abs(yeval_dd-fex)
    # plt.semilogy(xeval,err_l,'ro--',label='lagrange')
    plt.semilogy(xeval,err_dd,'bs--')
    plt.legend()
    plt.show()



def bary(xeval, xint, yint, N):
    wj = np.ones(N+1)
    
    for count in range(N+1):
        for j in range(N+1):
            if count != j:
                wj[count] = wj[count]/(xint[count]-xint[j])
    
    #eval
    lx = 1
    yeval = 0
    for i in range(N+1):
        # //eval l 
        lx *= (xeval - xint[i])
        
        # do formula
        yeval += yint[i]*wj[i] / (xeval - xint[i])
    
    return yeval * lx

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
  
    


''' create divided difference matrix'''
def dividedDiffTable(x, y, n):
 
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) / (x[j] - x[i + j]))
    return y
    
def evalDDpoly(xval, xint,y,N):
    ''' evaluate the polynomial terms'''
    ptmp = np.zeros(N+1)
    
    ptmp[0] = 1.
    for j in range(N):
      ptmp[j+1] = ptmp[j]*(xval-xint[j])
     
    '''evaluate the divided difference polynomial'''
    yeval = 0.
    for j in range(N+1):
       yeval = yeval + y[0][j]*ptmp[j]  

    return yeval



driver()        
# print(a)
# # print(np.cos(a*np.pi))
# xint = np.array([1 + np.cos(np.pi * (2*j - 1)/2*N) for j in range(1,N+1)])
# print(xint)