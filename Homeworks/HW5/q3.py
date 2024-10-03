
import numpy as np
import math
import time
from numpy.linalg import inv
from numpy.linalg import norm

def driver():
    q3()
    
def q3():
    x0 = np.array([1,1,1])
    Nmax = 100
    tol = 1e-10
    t = time.time()
    for j in range(50):
        [xstar,ier,its, iters] = Newton(x0,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('Newton: the error message reads:',ier)
    print('Newton: took this many seconds:',elapsed/50)
    print('Netwon: number of iterations is:',its)
    a = []
    b = []
    c=[]
    for i in iters:
        a.append(i[0])
        b.append(i[1])
        c.append(i[2])
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    compute_order(a, xstar[0])
    compute_order(b, xstar[1])
    compute_order(c, xstar[2])
    # t = time.time()
    # for j in range(20):
    #     [xstar,ier,its, iters] = LazyNewton(x0,tol,Nmax)
    # elapsed = time.time()-t
    # print(xstar)
    # print('Lazy Newton: the error message reads:',ier)
    # print('Lazy Newton: took this many seconds:',elapsed/20)
    # print('Lazy Newton: number of iterations is:',its)
    # a = []
    # b = []
    # for i in iters:
    #     a.append(i[0])
    #     b.append(i[1])
    # a = np.array(a)
    # b = np.array(b)
    
    # compute_order(a, xstar[0])
    # compute_order(b, xstar[1])
    
    # t = time.time()
    # for j in range(20):
    #     [xstar,ier,its] = Broyden(x0, tol,Nmax)
    # elapsed = time.time()-t
    # print(xstar)
    # print('Broyden: the error message reads:',ier)
    # print('Broyden: took this many seconds:',elapsed/20)
    # print('Broyden: number of iterations is:',its)
def f(x,y,z):
    return x**2 + 4*y**2 + 4*z**2 - 16
def fx(x):
    return 2*x
def fy(y):
    return 8 * y
def fz(z):
    return 8 * z
def evalF(x):
    F = np.zeros(3)
    F[0] = x[0] - f(x[0],x[1],x[2]) * fx(x[0]) / (fx(x[0])**2 + fy(x[1])**2 + fz(x[2])**2)
    F[1] = x[1] - f(x[0],x[1],x[2]) * fy(x[1]) / (fx(x[0])**2 + fy(x[1])**2 + fz(x[2])**2)
    F[2] = x[2] - f(x[0],x[1],x[2]) * fz(x[2]) / (fx(x[0])**2 + fy(x[1])**2 + fz(x[2])**2)
    return F       
       
def evalJ(x):
    J = np.array([[6*x[0], -2 * x[1]],
    [3*x[0]**2 - 3* x[1]**2, 6*x[0]*x[1]]])
    return J      
       
def compute_order(x, xstar):
    diff1 = np.abs(x[1::]-xstar)
    diff2 = np.abs(x[0:-1]-xstar)

    fit = np.polyfit(np.log(diff2.flatten()),np.log(diff1.flatten()),1)
    
    _lambda = np.exp(fit[1])
    alpha = fit[0]
    print(f"lambda: {_lambda}")
    # print(f"alpha: {alpha}")
    return fit       

       
def Newton(x0,tol,Nmax):
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''
    iterations = []
    for its in range(Nmax):
        iterations.append(x0)
        # J = evalJ(x0)
        # Jinv = inv(J)
        # F = evalF(x0)
        x1 = evalF(x0)
        if (norm(x1-x0) < tol):
            xstar = x1
            ier =0
            return[xstar, ier, its, iterations]
        x0 = x1
    xstar = x1
    ier = 1
    return[xstar,ier,its]       

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
    return[xstar,ier,its]
        
def Broyden(x0,tol,Nmax):
    '''tol = desired accuracy
    Nmax = max number of iterations'''
    '''Sherman-Morrison
    (A+xy^T)^{-1} = A^{-1}-1/p*(A^{-1}xy^TA^{-1})
    where p = 1+y^TA^{-1}Ax'''
    '''In Newton
    x_k+1 = xk -(G(x_k))^{-1}*F(x_k)'''
    '''In Broyden
    x = [F(xk)-F(xk-1)-\hat{G}_k-1(xk-xk-1)
    y = x_k-x_k-1/||x_k-x_k-1||^2'''
    ''' implemented as in equation (10.16) on page 650 of text'''
    '''initialize with 1 newton step'''
    A0 = evalJ(x0)
    v = evalF(x0)
    A = np.linalg.inv(A0)
    s = -A.dot(v)
    xk = x0+s
    for its in range(Nmax):
        '''(save v from previous step)'''
        w = v
        ''' create new v'''
        v = evalF(xk)
        '''y_k = F(xk)-F(xk-1)'''
        y = v-w
        '''-A_{k-1}^{-1}y_k'''
        z = -A.dot(y)
        ''' p = s_k^tA_{k-1}^{-1}y_k'''
        p = -np.dot(s,z)
        u = np.dot(s,A)
        ''' A = A_k^{-1} via Morrison formula'''
        tmp = s+z
        tmp2 = np.outer(tmp,u)
        A = A+1./p*tmp2
        ''' -A_k^{-1}F(x_k)'''
        s = -A.dot(v)
        xk = xk+s
        if (norm(s)<tol):
            alpha = xk
            ier = 0
            return[alpha,ier,its]
    alpha = xk
    ier = 1
    return[alpha,ier,its]


driver()