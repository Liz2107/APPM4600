import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def driver():
    # q1a()
    # q1b()
    q1c()
    # q4i()
    # q4iii()
    # q4ii()
    # q4_like_2c()
    # q5()
    
def q1c():
    x0 = 10
    tol = 1e-10
    T = lambda x: 35 * sp.special.erf(x / 1.69161697792) - 15
    Tp = lambda x: 70 * np.exp(-(x / 1.69161697792)**2) / np.sqrt(np.pi)
    [astar,ier, iters, count] = newton(T, Tp, x0, tol, 500)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', T(astar))
    print(f'took {count} iterations')
        
def q4_like_2c():
    print(x)
    
def q5():
    f = lambda x: x**6 - x - 1
    fp = lambda x: 6*x**5 - 1
    x0 = 2
    x1 = 1
    Nmax = 500
    tol = 1e-10
    
    [astar,ier, count, iters] = newton(f, fp, x0, tol, Nmax)
    # [astar, ier, count, iters] = secant(f, x0, x1, tol, Nmax)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print(f'took {count} iterations')
    
    errs = []
    for i in iters:
        errs.append(np.abs(i - astar))
    print(errs)
    
    iters = np.array(iters)
    newton_iters = np.array([0.8652758615984806, 0.5459041338497894, 0.296014849837543, 0.12024681770791701, 0.026814294371793723, 0.0016291357689859343, 6.389942109663593e-06, 9.870171346904044e-11])
    diff1n = np.abs(newton_iters[1::]-astar)
    diff2n = np.abs(newton_iters[0:-1]-astar)    
    
    
    diff1 = np.abs(iters[1::]-astar)
    diff2 = np.abs(iters[0:-1]-astar)
    
    plt.plot(diff1, diff2, label="Secant Method")
    # plt.plot(diff1n, diff2n, label="Newton's Method")
    plt.xscale("log")
    plt.yscale("log")
    # plt.legend()
    
    plt.xlabel("x_{k+1}")
    plt.ylabel("x_k")
    plt.title("Newton's method Logarithmic Convergence Visualizaiton")
    plt.show()    
    

def q4ii():
    f = lambda x: np.exp(3*x) - 27*x**6 + 27*x**4*np.exp(x) - 9*x**2*np.exp(2*x)
    fp = lambda x: 3 * np.exp(3*x) - 6*27*x**5 + 27*x**4*np.exp(x) + 4 * 27 * x**3 * np.exp(x) - 18 * x**2 * np.exp(2*x) - 18 * x * np.exp(2*x)
    fpp = lambda x: 9 * np.exp(3 * x) - - 30 * 27 * x**4 + 27*4*x**3*np.exp(x) + 27*x**4*np.exp(x) + 12 *27*x**2*np.exp(x) + 4 * 27 * x**3 * np.exp(x) -36 *x * np.exp(2*x) - 36 *x**2 * np.exp(2*x) - 18 * np.exp(2 * x) - 36 * x * np.exp(2 * x)

# set m = 3 (highest mult)
    g = lambda x: x - 3 * f(x) / fp(x)
    gp = lambda x: 6 *f(x) * fpp(x) / fp(x)**2
    x0 = 4
    tol = 1e-10
    Nmax = 500
    
    [astar,ier, count, iters] = newton(g, gp, x0, tol, Nmax)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print(f'took {count} iterations')
    
    
def q4iii():
    f = lambda x: np.exp(3*x) - 27*x**6 + 27*x**4*np.exp(x) - 9*x**2*np.exp(2*x)
    fp = lambda x: 3 * np.exp(3*x) - 6*27*x**5 + 27*x**4*np.exp(x) + 4 * 27 * x**3 * np.exp(x) - 18 * x**2 * np.exp(2*x) - 18 * x * np.exp(2*x)
    
    g = lambda x: f(x) / fp(x)
    fpp = lambda x: 9 * np.exp(3 * x) - - 30 * 27 * x**4 + 27*4*x**3*np.exp(x) + 27*x**4*np.exp(x) + 12 *27*x**2*np.exp(x) + 4 * 27 * x**3 * np.exp(x) -36 *x * np.exp(2*x) - 36 *x**2 * np.exp(2*x) - 18 * np.exp(2 * x) - 36 * x * np.exp(2 * x)
    gp = lambda x: 1 - f(x) * fpp(x) / fp(x)**2
    
    x0 = 4
    tol = 1e-10
    Nmax = 500
    
    [astar,ier, count, iters] = newton(g, gp, x0, tol, Nmax)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print(f'took {count} iterations')
    
def q4i():
    f = lambda x: np.exp(3*x) - 27*x**6 + 27*x**4*np.exp(x) - 9*x**2*np.exp(2*x)
    fp = lambda x: 3 * np.exp(3*x) - 6*27*x**5 + 27*x**4*np.exp(x) + 4 * 27 * x**3 * np.exp(x) - 18 * x**2 * np.exp(2*x) - 18 * x * np.exp(2*x)
    
    x0 = 4
    tol = 1e-10
    Nmax = 500
    
    [astar,ier, count, iters] = newton(f, fp, x0, tol, Nmax)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print(f'took {count} iterations')
        
        
def q1a():
    T = lambda x: 35 * sp.special.erf(x / 1.69161697792) - 15
    t = np.arange(0,10,0.05)
    plt.plot(t, [T(x) for x in t])
    plt.xlabel("x")
    plt.ylabel("T(x)")
    plt.title("Root finding of water main cooling function")
    plt.show()
    
    
def q1b():
    a = 0
    b = 10
    tol = 1e-10
    T = lambda x: 35 * sp.special.erf(x / 1.69161697792) - 15
    
    [astar,ier, count] = bisection(T,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', T(astar))
    print(f'took {count} iterations')
   
def secant(f, x0, x1, tol, Nmax):
    if np.abs(f(x0) == f(x1)):
        ier = 1
        xstar = x0
        return [xstar, ier, 0, []]
    count = 0
    iters = []
    for j in range(Nmax):
        iters.append(x1)
        count += 1
        x2 = x1 - f(x1)*(x1 - x0) / (f(x1) - f(x0))
        if np.abs(x2 - x1) < tol: #could make relative by adding over x2
            ier = 0
            xstar = x2
            return [xstar, ier, count, iters]
        x0 = x1
        x1 = x2
        if np.abs(f(x1)-f(x0)) == 0:
            ier = 1
            xstar = x1
            return [xstar, ier, count, iters]
    xstar = x2
    ier = 1
    return [xstar, ier, count, iters]   
   
    
def newton(f, fp, x0, tol, Nmax):
    if fp(x0) == 0:
        xstar = x0
        ier = 1
        return [xstar, ier, 0, []]
    count = 0
    iters = []
    for j in range(Nmax):
        x1 = x0 - f(x0) / fp(x0)
        iters.append(x0)
        count += 1
        if np.abs(x1 - x0) < tol:
            ier = 0
            xstar = x1
            return [xstar, ier, count, iters]
        x0 = x1
        if fp(x0) == 0:
            xstar = x0
            ier = 1
            return [xstar, ier, count, iters]
    ier = 1
    xstar = x1
    return [xstar, ier, count, iters]
    
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