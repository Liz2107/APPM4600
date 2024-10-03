import numpy as np
import time
from numpy.linalg import inv
from numpy.linalg import norm


def evalF(x):
    F = np.zeros(2)
    F[0] = 4*x[0]**2 + x[1]**2 - 4
    F[1] = x[0] + x[1] - np.sin(x[0] - x[1])
    return F       

def cd(s, h_, f):
    res = []
    for h in h_:
        res.append((f(s + h)-f(s-h))/2*h)
    return res

def approxJ(x, h):
    f = lambda x0,x1: 4*x0 + x1 
    g = lambda x0,x1: x0 + x1 -np.sin(x0 - x1)
    return np.array([[(f(x[0] + h*x[0],x[1])-f(x[0]-h,x[1]))/2*h, (f(x[0],x[1]+h)-f(x[0],x[1]-h))/2*h],[(g(x[0] + h,x[1])-g(x[0]-h,x[1]))/2*h, (g(x[0],x[1]+h)-g(x[0],x[1]-h))/2*h]])
       
def evalJ(x):
    J = np.array([[8*x[0], 2* x[1]],
    [1 - np.cos(x[0] - x[1]), 1 + np.cos(x[0] - x[1])]])
    return J      
       
def compute_order(x, xstar):
    diff1 = np.abs(x[1::]-xstar)
    diff2 = np.abs(x[0:-1]-xstar)

    fit = np.polyfit(np.log(diff2.flatten()),np.log(diff1.flatten()),1)
    
    _lambda = np.exp(fit[1])
    alpha = fit[0]
    print(f"lambda: {_lambda}")
    print(f"alpha: {alpha}")
    return fit       

       
def Slackernewton(x0, tol, Nmax):
    iters = []
    J = evalJ(x0)
    Jinv = inv(J)
    for its in range(Nmax):
        iters.append(x0)
        F = evalF(x0)
        x1 = x0 - Jinv.dot(F)
        if norm(x1-x0) < tol:
            xstar = x1
            ier = 0
            return [xstar, ier, its, iters]
        if len(iters) > 1 and np.abs((iters[-1][0] - iters[-2][0])) > .001:
            J = evalJ(x0)
            Jinv = inv(J)
        x0 = x1
    xstar = x1
    ier = 1
    return [xstar, ier, its, iters]
       
def Newton(x0,tol,Nmax):
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''
    iterations = []
    for its in range(Nmax):
        iterations.append(x0)
        J = approxJ(x0, 1e-7)
        Jinv = inv(J)
        F = evalF(x0)
        x1 = x0 - Jinv.dot(F)
        if (norm(x1-x0) < tol):
            xstar = x1
            ier =0
            return[xstar, ier, its, iterations]
        x0 = x1
    xstar = x1
    ier = 1
    return[xstar,ier,its, iterations]       

def LazyNewton(x0,tol,Nmax):
    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''
    iterations = []
    J = evalJ(x0)
    Jinv = inv(J)
    for its in range(Nmax):
        iterations.append(x0)
        F = evalF(x0)
        x1 = x0 - Jinv.dot(F)                
        if (norm(x1-x0) < tol):
            xstar = x1
            ier =0
            return[xstar, ier,its, iterations]
        x0 = x1
    xstar = x1
    ier = 1
    return[xstar,ier,its,iterations]


x0 = np.array([1,0])
Nmax = 1000
tol = 1e-10
# t = time.time()
# for j in range(50):
#     [xstar,ier,its, iters] = LazyNewton(x0,tol,Nmax)
# elapsed = time.time()-t
# print(xstar)
# print('Newton: the error message reads:',ier)
# print('Newton: took this many seconds:',elapsed/50)
# print('Netwon: number of iterations is:',its)
# a = []
# b = []
# for i in iters:
#     a.append(i[0])
#     b.append(i[1])
# a = np.array(a)
# b = np.array(b)

# compute_order(a, xstar[0])
# compute_order(b, xstar[1])

# for j in range(50):
#     [xstar,ier,its, iters] = Slackernewton(x0,tol,Nmax)
# elapsed = time.time()-t
# print(xstar)
# print('S Newton: the error message reads:',ier)
# print('Newton: took this many seconds:',elapsed/50)
# print('Netwon: number of iterations is:',its)
# a = []
# b = []
# for i in iters:
#     a.append(i[0])
#     b.append(i[1])
# a = np.array(a)
# b = np.array(b)

# compute_order(a, xstar[0])
# compute_order(b, xstar[1])
for j in range(50):
    [xstar,ier,its, iters] = Newton(x0,tol,Nmax)
# elapsed = time.time()-t
print(xstar)
print('S Newton: the error message reads:',ier)
# print('Newton: took this many seconds:',elapsed/50)
print('Netwon: number of iterations is:',its)
a = []
b = []
for i in iters:
    a.append(i[0])
    b.append(i[1])
a = np.array(a)
b = np.array(b)

compute_order(a, xstar[0])
compute_order(b, xstar[1])