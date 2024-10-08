import matplotlib.pyplot as plt
import numpy as np

def driver():
    # q5a()
    # q1()
    # q2()
    # q3()
    q5b()
    
def q5b():
    # f = lambda x: -np.sin(2*x) + 5*x/4 - 3/4
    # x0 = 1.5
    f = lambda x: -np.sin(2*x) + 5*x/4 - 3/4
    x0 = 3
    Nmax = 1000
    tol = 1e-10
    
    [xstar,ier] = fixedpt(f,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('f3(xstar):',f(xstar))
    print('Error message reads:',ier)
    
def q3():
    f = lambda x: x**3 + x - 4
    a = 1
    b = 4
    tol = 1e-3
    [astar,ier, count] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print(f'took {count} iterations')
    
def q5a():
    t = np.arange(-3, 9, 0.01)
    y = [x - 4*np.sin(2*x) - 3 for x in t]
    plt.plot(t, y)
    plt.plot([-3,9],[0,0])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Roots of x - 4sin(2x) - 3")
    plt.show()
    return

def q1():
    f = lambda x: -2 * x + 1 + np.sin(x)
    tol = 1e-8
    a = 0
    b = 1
    
    [astar,ier, count] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print(f'took {count} iterations')
    
def q2():
    f1 = lambda x: (x-5)**9
    f2 = lambda x: x**9 - 45*x**8 + 900*x**7 - 10500*x**6 + 78750*x**5 - 393750*x**4 + 1312500*x**3 - 2812500*x**2 + 3515625*x - 1953125
    tol = 1e-4
    a = 4.82
    b = 5.2
    
    print("Non-Expanded")
    [astar,ier, count] = bisection(f1,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f1(astar))
    print(f'took {count} iterations')
    
    print("Expanded")
    [astar,ier, count] = bisection(f2,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f2(astar))
    print(f'took {count} iterations')
    
def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0
    while (count <Nmax):
       count = count +1
       x1 = f(x0)
       if (abs(x1-x0) <tol):
          xstar = x1
          ier = 0
          return [xstar,ier]
       x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier]

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