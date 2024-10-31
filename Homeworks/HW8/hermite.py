import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import math

def make_nodes(N):
    a = []
    for i in range(N+1):
        a.append(5*np.cos(np.pi * (2*i+1) / (2*N)))
    a.reverse()
    return np.array(a)

def driver():
    for n in [6, 16,20]:
        print(make_nodes(n))

    f = lambda x: 1/(1+x**2)
    fp = lambda x: -2*x/(1+x**2)**2

    N = 16
    ''' interval'''
    a = -5
    b = 5
   
    ''' create equispaced interpolation nodes'''
    xint = [-4.97592363, -4.78470168, -4.40960632, -3.86505227, -3.17196642,  -2.35698368, -1.45142339, -0.4900857,0,   0.4900857,   1.45142339,  2.35698368, 3.17196642,  3.86505227,  4.40960632,  4.78470168,  4.97592363]
    # xint = [-4.98458667, -4.8618496,  -4.61939766, -4.26320082, -3.80202983, -3.24724024, -2.61249282, -1.91341716, -1.16722682, -0.39229548,  0, 0.39229548, 1.16722682,  1.91341716,  2.61249282,  3.24724024,  3.80202983,  4.26320082,  4.61939766,  4.8618496,   4.98458667]
    ''' create interpolation data'''
    yint = np.zeros(N+1)
    ypint = np.zeros(N+1)
    for jj in range(N+1):
        yint[jj] = f(xint[jj])
        ypint[jj] = fp(xint[jj])
    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yevalL = np.zeros(Neval+1)
    yevalH = np.zeros(Neval+1)
    for kk in range(Neval+1):
      yevalH[kk] = eval_hermite(xeval[kk],xint,yint,ypint,N)

    ''' create vector with exact values'''
    fex = np.zeros(Neval+1)
    for kk in range(Neval+1):
        fex[kk] = f(xeval[kk])
    
    
    plt.figure()
    plt.plot(xeval,fex,'ro-')
    plt.plot(xeval,yevalH,'c.--',label='Hermite')
    plt.semilogy()
    plt.show()
         
    errL = abs(yevalL-fex)
    errH = abs(yevalH-fex)
    plt.figure()
    plt.semilogy(xeval,errH,'c.--',label='Hermite')
    plt.show()            


def eval_hermite(xeval,xint,yint,ypint,N):

    ''' Evaluate all Lagrange polynomials'''

    lj = np.ones(N+1)
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    ''' Construct the l_j'(x_j)'''
    lpj = np.zeros(N+1)
#    lpj2 = np.ones(N+1)
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
#              lpj2[count] = lpj2[count]*(xint[count] - xint[jj])
              lpj[count] = lpj[count]+ 1./(xint[count] - xint[jj])
              
              

    yeval = 0.
    
    for jj in range(N+1):
       Qj = (1.-2.*(xeval-xint[jj])*lpj[jj])*lj[jj]**2
       Rj = (xeval-xint[jj])*lj[jj]**2
#       if (jj == 0):
#         print(Qj)
         
#         print(Rj)
#         print(Qj)
#         print(xeval)
 #        return
       yeval = yeval + yint[jj]*Qj+ypint[jj]*Rj
       
    return(yeval)
       


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
  
    

       
if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()        
