import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from scipy.fft import fft, fftfreq

mpl.rcParams.update({'font.size': 16})
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.linewidth'] = 2

mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.size'] = 2
mpl.rcParams['xtick.minor.width'] = 1

mpl.rcParams['ytick.major.size'] = 8
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.size'] = 2
mpl.rcParams['ytick.minor.width'] = 1

mpl.rcParams['xtick.major.pad']='8'
mpl.rcParams['ytick.major.pad']='8'

def driver():
    # times = 10, 50, 100, 250, 500, 1000, 5000, 10000
    times = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    dft = [make_dft(n) for n in times]
    fft_ = [make_fft(n) for n in times]
    # print(make_fft(1000000))
    # print(dft)
    # print(fft_)
    # make_dft(100)
    # make_fft(100)
    plt.plot(times, dft, 'tab:red', label="DFT")
    plt.plot(times, fft_, 'tab:blue', label="FFT")
    plt.title("Time Comparison of DFT vs FFT")
    plt.semilogx()
    # plt.semilogy()
    plt.xlabel("Number of Points in Signal")
    plt.ylabel("Time to complete (s)")
    plt.legend()
    plt.show()

def DFT(f):

    N = len(f)
    n = np.arange(N)
    k = []
    for i in n:
        k.append([i])
        
    k = np.array(k)
    
    exponential = np.exp(-2j * np.pi * k * n / N)
    
    res = np.dot(exponential, f)
    return res

def make_dft(pts):
    start = time.time()
    for i in range(10):
        ts = 1.0/pts
        t = np.arange(0,1,ts)

        f = 0.3*np.sin(2*np.pi*t) + 5*np.sin(2*np.pi*24*t) + 3* np.sin(2*np.pi*5*t)

        # ax = plt.subplot(1,2,1)
        # plt.plot(t, f, 'tab:red')
        # plt.ylabel('Amplitude')
        # plt.xlabel("Time")
        # ax.set_title("Original Signal")

        res = DFT(f)

        N = len(res)
        n = np.arange(N)
        T = N/pts
        freq = n/T 

    # ax2 = plt.subplot(1,2,2)
    # markerline, stemlines, baseline = plt.stem(freq[0:N//2], abs(res[0:N//2]), 'tab:blue', markerfmt=" ", basefmt="tab:blue")
    # # plt.plot(freq, [0 for i in freq], 'tab:blue')
    # plt.setp(stemlines, linewidth=3)
    # plt.xlabel('Frequency')
    # plt.ylabel('DFT')
    # ax2.set_title("Transform")
    # # plt.title("DFT")
    # plt.show()
    end = time.time()
    return (end - start) / 10

def make_fft(N):
    N *= 100
    start = time.time()
    for i in range(50):
        T = 1.0 / N
        x = np.linspace(0.0, N*T, N, endpoint=False)
        # y = 2*np.sin(4*np.pi*x) + 4*np.sin(2*np.pi*28*x) + .5* np.sin(2*np.pi*15*x)
        y = 0.3*np.sin(2*np.pi*x) + 5*np.sin(2*np.pi*24*x) + 3* np.sin(2*np.pi*5*x)
        yf = fft(y)

        xf = fftfreq(N, T)[:N//2]

    # ax = plt.subplot(1,2,1)
    # plt.plot(x, y, 'tab:red')
    # plt.ylabel('Amplitude')
    # plt.xlabel("Time")
    # ax.set_title("Original Signal")

    # ax2 = plt.subplot(1,2,2)
    # markerline, stemlines, baseline = plt.stem(xf, 2.0/N * np.abs(yf[0:N//2]), 'tab:blue', markerfmt=" ", basefmt="tab:blue")
    # # plt.plot(freq, [0 for i in freq], 'tab:blue')
    # plt.setp(stemlines, linewidth=3)
    # plt.xlabel('Frequency')
    # plt.ylabel('FFT')
    # ax2.set_title("Transform")
    # # plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    

    # plt.show()
    end = time.time()
    return (end - start) /50


driver()