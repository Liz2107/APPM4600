import matplotlib.pyplot as plt
import numpy as np

def driver():
    q3()
    
def q3():
    t = np.arange(-3, 5, 0.01)
    y = [x - 4*np.sin(2*x) - 3 for x in t]
    plt.plot(t, y)
    plt.plot([-4,-1,0,1,2,3,4,6],[0,0,0,0,0,0,0,0])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Roots of x - 4sin(2x) - 3")
    plt.show()
    return

driver()