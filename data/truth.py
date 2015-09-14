
import numpy as np
import matplotlib.pyplot as plt

A = np.array([-0.21859758,  1.09006246, -0.2406215 ,  0.52241687, -3.42709706,
               0.73309541,  1.15969302,  0.7274046 ,  0.65796667,  0.14310152])
R = np.array([-0.16898475, -0.90798292,  0.71474506,  0.60564753, -0.05457973,
               0.08473391, -0.81566986,  0.09358501,  0.34645654,  0.56111046])/3.0
xo = np.array([ 0.24653083,  0.98569815,  0.56011908,  0.90489527,  0.05984143,
                0.82936614,  0.8839411 ,  0.6548733 ,  0.20827336,  0.74536993])
                

def truth(x):
    y = np.sin(4*np.pi*x)
    for n in range(len(A)):
        y += A[n] * np.exp( -(x-xo[n])**2 / R[n]**2)
    return y
    


x = np.random.rand(50)
y = truth(x) + 0.1*np.random.randn(len(x))

print 'saving..'
np.savetxt('oi_data.dat', np.vstack((x, y)).T)


plt.plot(x, truth(x), 'r+')
plt.show()

