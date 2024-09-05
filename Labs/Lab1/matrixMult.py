import numpy as np
import numpy.linalg as la
import time
def driver():
    A = np.array([[1,2,-1],[3,4,-1]])
    b = np.array([3,-1,0])
    
    my_t = 0
    start = time.time()
    for i in range(10000):
        matrixMultiply(A, 2, 2, b)
    end = time.time()
    my_t = end - start
    
    np_t = 0
    start_ = time.time()
    for i in range(10000):
        np.matmul(A, b)
    end_ = time.time()
    np_t = end_ - start_
    
    # print the output
    print('The product computed by the program is : ', matrixMultiply(A,2,2,b))
    print('The produce computed by numpy is ', np.matmul(A, b))
    print(f'On average, my code took {(my_t):.5f} s to run 10,000 iterations, and numpy took {(np_t /10000):.10f} s to run 10,000 iterations.')
    
    return
def dotProduct(x,y,n):
    # Computes the dot product of the n x 1 vectors x and y
    dp = 0.
    for j in range(n):
        dp = dp + x[j]*y[j]
    return dp
def matrixMultiply(A,m,n,b):
    result = np.zeros(m)
    for i in range(n):
        result[i] = dotProduct(A[i],b,n)
    return result


driver()


