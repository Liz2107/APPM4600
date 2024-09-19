import numpy as np
import matplotlib.pyplot as plt
import random
import math

# 3 

def f(x):
    y = math.exp(x)
    return y - 1

# check stability and visualize:
t = np.arange(-5,5,0.01)
plt.plot(t, [f(x) for x in t])
plt.show()

#digits of precision:
x = 9.999999995000000 * 10**-10
print(f(x))

#ts approach
value = 0
for i in range(1,15):
    value += x**i / math.factorial(i)
    # print(f"On the addition of the {i}th term, the result was {value}")
    # print(value - 10**-9)
print(x + x**2/2) # outputs 10^-9


# 4 (a)
t = []
y = []

for i in range(1,31):
    x = np.pi * i / 30
    t.append(x)
    y.append(np.cos(x))

t = np.array(t)
y = np.array(y)

N = 30

res = 0

for i in range(N):
    res += t[i] * y[i]

print("the sum is ", res)    

# 4 (b)

def x(theta, R, dr, f, p):
    return R * np.cos(theta) * (1 + dr * np.sin(f * theta + p))
def y(theta, R, dr, f, p):
    return R * np.sin(theta) * (1 + dr * np.sin(f * theta + p))

R = 1.2
dr = 0.1
f = 15
p = 0

t = np.arange(0, 2 * np.pi, 0.01)

x_ = [x(t_, R, dr, f, p) for t_ in t]
y_ = [y(t_, R, dr, f, p) for t_ in t]


fig1, ax = plt.subplots(2,1,figsize=(5,10))
ax[0].plot(x_, y_)
ax[0].set_title("Wavy Circles Plot #1")
ax[0].set_box_aspect(1)

dr = 0.05
for i in range(10):
    R = i + 1
    f = i + 3
    p = random.uniform(0,2)
    ax[1].plot([x(t_, R, dr, f, p) for t_ in t], [y(t_, R, dr, f, p) for t_ in t], label=i)
ax[1].set_title("Wavy Circles Plot #2")
ax[1].set_box_aspect(1)

# plt.show()

 