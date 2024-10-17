import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 


def driver():
    
    f = lambda x: math.exp(x)
    a = 0
    b = 1
    
    ''' create points you want to evaluate at'''
    Neval = 100
    xeval =  np.linspace(a,b,Neval)
    
    ''' number of intervals'''
    Nint = 2
    
    '''evaluate the linear spline'''
    yeval = eval_lin_spline(xeval,Neval,a,b,f,Nint)
    
    ''' evaluate f at the evaluation points'''
    fex = np.zeros(Neval)
    for j in range(Neval):
      fex[j] = f(xeval[j]) 
      
    plt.figure()
    plt.plot(xeval,fex,label="Actual Function")
    plt.plot(xeval,yeval, label="Evaluated Function")
    plt.legend()
    plt.show()   
     
    plt.figure()
    
    err = abs(yeval-fex)
    plt.plot(xeval, err)
    plt.show()            

    
def line_eval(a, b, fa, fb, x):
  m = (fa - fb)/(a-b)
  return fa + m*(x - a)


def  eval_lin_spline(xeval,Neval,a,b,f,Nint):

    '''create the intervals for piecewise approximations'''
    xint = np.linspace(a,b,Nint+1)
    # print(xint)
    '''create vector to store the evaluation of the linear splines'''
    yeval = np.zeros(Neval) 

    for jint in range(Nint):
      # print(xint[jint])
      a1= xint[jint]
      fa1 = f(a1)
      b1 = xint[jint+1]
      fb1 = f(b1)
      ind = []
      count = 0
      for x in xeval:
        if x >= a1 and x <= b1:
          ind.append(x)
          yeval[count] = line_eval(a1, b1, fa1, fb1, x)
          # print(a1, " to ", b1, "x:", x, "y: ", yeval[count])
        count += 1
          
      # n = len(ind)
      
      # for kk in ind:
        # yeval[kk] = line_eval(a, b, fa1, fb1, kk)
          
          #  '''use your line evaluator to evaluate the lines at each of the points 
          #  in the interval'''
          #  '''yeval(ind(kk)) = call your line evaluator at xeval(ind(kk)) with 
          #  the points (a1,fa1) and (b1,fb1)'''
    return yeval
           
           
if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()               
