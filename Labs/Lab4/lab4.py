import numpy as np



def driver():
    f1 = lambda x: (10 / (x + 4))**0.5
    tol = 1e-10
    Nmax=300
    x0 = 1.5
    [xstar,ier,iters] = fixedpt(f1,x0,tol,Nmax)
    # print('the approximate fixed point is:',xstar)
    # print('f1(xstar):',f1(xstar))
    # print('Error message reads:',ier)
    # print(iters)
    
    # compute_order(iters, xstar)
    
    # p = new_seq(iters, tol, Nmax)
    # # print(p)
    
    # compute_order(np.array(p), xstar)
    
    x_, iters, ier = steff(f1, 1.5, Nmax, tol)
    print(x_)
    print(iters)
    
    return

def compute_order(x, xstar):
    diff1 = np.abs(x[1::]-xstar)
    diff2 = np.abs(x[0:-1]-xstar)

    fit = np.polyfit(np.log(diff2.flatten()),np.log(diff1.flatten()),1)
    
    _lambda = np.exp(fit[1])
    alpha = fit[0]
    print(f"lambda: {_lambda}")
    print(f"alpha: {alpha}")
    return fit



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
        #   iters.append(x1)
          xstar = x1
          ier = 0
          return [xstar,ier,np.array(iters)]
       iters.append(x0)
       x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, np.array(iters)]
    
def steff(g, p0, nmax, tol):
    count = 0
    iterations = []
    while count < nmax:
        count += 1
        iterations.append(p0)
        p1 = p0 - (g(p0) - p0)**2/(g(g(p0)) - 2*g(p0) + p0)
        if p1 -p0 < 1e-7:
            print(count)
            return [p1, iterations, 0]
        p0 = p1
    return [p0, iterations, 1]


def new_seq(p, tol, Nmax):
    new_iters = []
    count = 0
    while (count < p.size - 3 and count < Nmax):
        count += 1
        # p = (iters[count]*iters[count+2] - iters[count+1]**2)/(-2*iters[count+1]+iters[count]+iters[count+2]) 
        p_hat = p[count] - (p[count+1]-p[count])**2/(p[count+2]-2*p[count+1]+p[count])
        new_iters.append(p_hat)
        if count > 1 and np.abs(p_hat - new_iters[-2] < tol):
            return new_iters
    return new_iters   
        
        
driver()