#Jessica Marshall
#ECE414 Machine Learning
#Linear Regression Assignment
#recreate figure 3.8

##########################################
#import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm

##########################################
#constants
N = [1, 2, 4, 25]

xn = np.random.uniform(0, 1, N[3])
y_synthetic = np.sin(2*np.pi*xn)
noise = np.random.normal(0, 0.15, N[3])
yn = y_synthetic + noise
#plt.scatter(xn, yn)

mu = np.linspace(0, 1, 9)       #define mus of radial Gaussian basis functions
s = 1                           #for now
#for N = 1
#iota is a 1x9 matrix (1 observation by 9 basis functions)
