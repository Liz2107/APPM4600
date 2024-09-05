import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

#Q1 Code
# x = np.arange(1.92,2.08,.001)

# p1 = lambda x: x**9 - 18*x**8 + 144*x**7 - 672*x**6 + 2016*x**5 - 4032*x**4 + 5376*x**3 - 4608*x**2 + 2304*x - 512
# p2 = lambda x: (x-2)**9

# y1 = p1(x)
# y2 = p2(x)

# # plt.plot(x, y1)
# plt.plot(x,y2)
# plt.xlabel("x")
# plt.ylabel("p(x)")
# # plt.title("p(x) = x^9 -18x^8 +144x^7 -672x^6 +2016x^5 -4032x^4 +5376x^3 -4608x2 +2304x-512 = (x-2)^9")
# plt.title("p(x) - (x-2)^9")

# plt.show()

#Question 5 b
x1 = np.pi
x2 = 106
f1 = lambda d: -2 * np.sin((2*x1 + d)/2) * np.sin((d)/2)
f2 = lambda d: -2 * np.sin((2*x2 + d)/2) * np.sin((d)/2)

g1 = lambda d: np.cos(x1 + d) - np.cos(x1)
g2 = lambda d: np.cos(x2 + d) - np.cos(x2)

# y1 = []
# y2 = []
# d = []
# for i in range(1, 18):
#     factor = 10 ** i
#     delta = 10**(-17) * factor 
#     d.append(delta)
#     y1.append(abs(f1(delta) - g1(delta)))
#     y2.append(abs(f2(delta) - g2(delta)))
    
# plt.plot(d, y2, label="x=106")
# plt.plot(d, y1, label="x=pi")

# plt.title("Comparitive errrors")
# plt.xlabel("Logarithmic Delta")
# plt.ylabel("Logarithmic Error")
# plt.xscale("log")
# plt.yscale("log")
# plt.legend()
# plt.show()

# plt.savefig("error_comp_delta_x")

# Question 5c

t1 = lambda d: d * -np.sin(x1) + (d**2 / 2) * -np.cos(x1+d/2) 
t2 = lambda d: d * -np.sin(x2) + (d**2 / 2) * -np.cos(x2+d/2) 

y1 = []
y2 = []
d = []
for i in range(1, 18):
    factor = 10 ** i
    delta = 10**(-17) * factor 
    d.append(delta)
    y1.append(abs(t1(delta) - g1(delta)))
    y2.append(abs(t2(delta) - g2(delta)))
    
plt.plot(d, y2, label="x=106")
plt.plot(d, y1, label="x=pi")

plt.title("Comparitive errrors")
plt.xlabel("Logarithmic Delta")
plt.ylabel("Logarithmic Error")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()

plt.savefig("error_comp_delta_x")