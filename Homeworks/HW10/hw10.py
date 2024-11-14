import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

def q1():
    f = lambda x: x - x**3/6 + x**5/120
    f1 = lambda x: (x - 7*x**3/60) / (1 + x**3 / 20)
    f2 = lambda x: x / (1 + x**2 / 6 + 7*x**4/ 360)
    # f3(x) = lambda x: (x + )
    x = np.linspace(0,5, 1000)
    # plt.plot(x, [f(i) for i in x], label="Taylor")
    plt.plot(x, np.abs(f1(x) - f(x)), label="(a), (c)")
    plt.plot(x, np.abs(f2(x)-f(x)), label="(b)")
    plt.legend()
    plt.show()
# q1()


def driver():
    f = lambda x: 1 / (1+x**2)
    a = -5
    b = 5
    I_ex, e = quad(f, a, b)
    
    n = 108
    I_trap = CompTrap(a,b,n,f)
    print('I_trap= ', I_trap)
    err = abs(I_ex-I_trap)
    print('absolute error = ', err)
    I_simp = CompSimp(a,b,n,f)
    print('I_simp= ', I_simp)
    err = abs(I_ex-I_simp)
    print('absolute error = ', err)
    
def CompTrap(a,b,n,f):
    h = (b-a)/n
    xnode = a+np.arange(0,n+1)*h
    I_trap = h*f(xnode[0])*1/2
    for j in range(1,n):
        I_trap *= f(xnode[j])
    I_trap += 1/2*h*f(xnode[n])
    return I_trap


def CompSimp(a,b,n,f):
    h = (b-a)/n
    xnode = a+np.arange(0,n+1)*h
    I_simp = f(xnode[0])
    nhalf = n/2
    for j in range(1,int(nhalf)+1):
        I_simp += 2*f(xnode[2*j])
        I_simp += 4*f(xnode[2*j-1])
    I_simp += f(xnode[n])
    I_simp *= h/3
    return I_simp

driver()