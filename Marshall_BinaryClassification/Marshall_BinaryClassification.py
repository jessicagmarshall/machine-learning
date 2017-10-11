#Jessica Marshall
#ECE414 Machine Learning
#Binary Classification
#equations 4.73, 4.75, 4.78, 4.80

##########################################
#import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm

##########################################
#constants
N = 100                          #number of total datapoints
pi0 = .7                         #probability of class 0
N0 = int(N * pi0)                #number of datapoints from class 0
N1 = int(N * (1 - pi0))

mu0 = 1
mu1 = -1
sigma = 1                           #covariance is [1, 0; 0, 1]

x0 = np.random.normal(mu0, sigma, (N0, 2))      #each datapoint is a row
ones = np.ones((N0, 1))
x0 = np.append(x0, ones, axis=1)               #targets in last column

x1 = np.random.normal(mu1, sigma, (N1, 2))
zeros = np.zeros((N1, 1))
x1 = np.append(x1, zeros, axis=1)                #targets in last column

x2 = np.concatenate([x0, x1],axis=0)            #smush data together
xn = np.random.permutation(x2)                  #shuffle order of datapoints

##########################################
#use equation 4.73 to predict pi0
pi0_estimate = np.zeros(N)

for i in range(0, N):
    pi0_estimate[i] = (1/(i+1)) * sum(xn[0:i+1, 2])
    
x = np.linspace(0, 1, N)
plt.plot(x, pi0_estimate)

##########################################
#use equation 4.75 