import numpy as np
import matplotlib.pyplot as plt

def driver():
    f = lambda x: np.exp(x**2 + 7 * x - 30) - 1
    fp = lambda x: (2*x + 7) * np.exp(x**2 + 7 * x - 30)
    fpp = lambda x: (2*x + 7)**2 * np.exp(x**2 + 7 * x - 30) + 2 * np.exp(x**2 + 7 * x - 30)
    
    tol = 1e-5
    Nmax = 300
    
    # bisection 
    # [astar,ier, count] = bisection(f,2,4.5,tol)
    # print('the approximate root is',astar)
    # print('the error message reads:',ier)
    # print('f(astar) =', f(astar))
    # print(f'took {count} iterations')
    
    # #newton
    # x0 = 4.5
    # [iters,astar,ier, count] = newton(f, fp, x0, tol, 500)
    # print('the approximate root is',astar)
    # print('the error message reads:',ier)
    # print('f(astar) =', f(astar))
    # print(f"took {count} iterations")

    # hybrid 
    [iters,astar,ier, count] = bisection_newton(f, fp, fpp, 2,4.5, 500, tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print(f"took {count} iterations")
def newton(f,fp,p0,tol,Nmax):
    p = np.zeros(Nmax+1)
    p[0] = p0
    for it in range(Nmax):
        p1 = p0-f(p0)/fp(p0)
        p[it+1] = p1
        if (abs(p1-p0) < tol):
            pstar = p1
            info = 0
            return [p,pstar,info,it]
        p0 = p1
    pstar = p1
    info = 1
    return [p,pstar,info,it]

def bisection_newton(f,fp, fpp, a,b, Nmax, tol):
    
#    Inputs:
#     f,a,b       - function and endpoints of initial interval
#      tol  - bisection stops when interval length < tol

#    Returns:
#      astar - approximation of root
#      ier   - error message
#            - ier = 1 => Failed
#            - ier = 0 == success

#     first verify there is a root we can find in the interval 
    # fp = lambda x: np.gradient(f(x),x)
    
    count = 0
    fa = f(a)
    fb = f(b)
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier, count]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier, count]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier, count]

    count = 0
    d = 0.5*(a+b)
    while abs(d-a) > f(d) * fpp(d) / fp(d)**2:
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier, count]
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
    print(count)
    return newton(f, fp, astar, tol, Nmax)

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
    count = 0
    fa = f(a)
    fb = f(b)
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier, count]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier, count]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier, count]

    count = 0
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier, count]
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
    return [astar, ier, count]

driver()