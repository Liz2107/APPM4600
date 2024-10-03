import numpy as np

f = lambda x: np.cos(x)

h = 0.01*2.**(-np.arange(0, 10))

s = np.pi / 2


def fd(s, h_):
    res = []
    for h in h_:
        res.append((f(s + h)-f(s))/h)
    return res


def cd(s, h_):
    res = []
    for h in h_:
        res.append((f(s + h)-f(s-h))/2*h)
    return res

def compute_order(x, xstar):
    diff1 = np.abs(x[1::]-xstar)
    diff2 = np.abs(x[0:-1]-xstar)

    fit = np.polyfit(np.log(diff2.flatten()),np.log(diff1.flatten()),1)
    
    _lambda = np.exp(fit[1])
    alpha = fit[0]
    print(f"lambda: {_lambda}")
    # print(f"alpha: {alpha}")
    return fit  

fd_r = fd(s, h)
print(fd_r)
compute_order(np.array(fd_r), -1)

cd_r = cd(s, h)
print(cd_r)
compute_order(np.array(cd_r), -1)