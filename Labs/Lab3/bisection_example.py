# import libraries
import numpy as np

def driver():

# use routines

    # f = lambda x: x**3 - x**2
    
    # 1 a
    # a = .5
    # b = 2
    # returns:
    # the approximate root is 0.9999999701976776
    # the error message reads: 0
    # f(astar) = -2.9802320611338473e-08
    # this works as normal

    # 1 b
    # a = -1
    # b = 0.5
    # returns
    # the approximate root is -1
    # the error message reads: 1
    # f(astar) = -2
    # This cannot find the root at zero since it has multiplicity and the positivity doesn't change
    
    # 1 c
    # a = -1
    # b = 2
    #returns
    # the approximate root is 0.9999999701976776
    # the error message reads: 0
    # f(astar) = -2.9802320611338473e-08
    # this finds the root at -1 since it can't do the zero root. The change in positivity works here



    # 2
    
    tol = 1e-5
    
    #2 a
    # f = lambda x: (x-1) * (x-3) * (x-5)
    # a = 0
    # b = 2.4
    #returns:
    #the approximate root is 1.0000030517578122
    # the error message reads: 0
    # f(astar) = 2.4414006618542327e-05
    # works as expected, finds the only root in range
    
    # 2 b
    f = lambda x: (x-1)**2 * (x-3) 
    a = 0
    b = 2
    #returns 
#     the approximate root is 0
    # the error message reads: 1
    # f(astar) = -3
    # this occurs because the root has multiplicity 2 > 1, so there is no change in sign
    
    # 2 c i
    f = lambda x: np.sin(x)
    # a = 0
    # b = .1
    # with these a and b it returns 0 for all values. This makes sense
    
    # 2 c ii
    a = 0.5
    b = 0.75 * np.pi
    # the approximate root is 0.5
    # the error message reads: 1
    # f(astar) = 0.479425538604203

# Yes, this code was expected given the 

    [astar,ier] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))




# define routines
def bisection(f,a,b,tol):
    
#    Inputs:
#     f,a,b       - function and endpoints of initial interval
#      tol  - bisection stops when interval length < tol

#    Returns:
#      astar - approximation of root
#      ier   - error message
#            - ier = 1 => Failed
#            - ier = 0 == success

#     first verify there is a root we can find in the interval 

    fa = f(a)
    fb = f(b);
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier]

    count = 0
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count +1
#      print('abs(d-a) = ', abs(d-a))
      
    astar = d
    ier = 0
    return [astar, ier]
      
driver()               

