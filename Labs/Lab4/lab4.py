import numpy as np



def driver():
    f1 = lambda x: x**2 - 3*x
    tol = 1e-7
    Nmax=30
    x0 = 2
    [xstar,ier,iters] = fixedpt(f1,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('f1(xstar):',f1(xstar))
    print('Error message reads:',ier)
    print(iters)
    
    
    return

# define routines
def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''
    iters = []
    count = 0
    while (count <Nmax):
       count = count +1
       x1 = f(x0)
       if (abs(x1-x0) <tol):
          iter.append(x1)
          xstar = x1
          ier = 0
          return [xstar,ier,np.array(iters)]
       iters.append(x1)
       x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, np.array(iters)]
    

driver()