#Jessica Marshall
#ECE414 Machine Learning
#Linear Regression Assignment
#recreate figure 3.7

##########################################
#import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

##########################################
#constants
N = 100                             #number of datapoints
beta_noise = 25                     #defined by the book
alpha = 2.0                         #defined by the book

a0 = -0.3                           #defined by the book
a1 = 0.5                            #defined by the book
xn = np.random.uniform(-1, 1, N)    #defined by the book
tn_synthetic = a0 + (a1*xn)         #generate synthetic data

noise = np.random.normal(0, 0.2, N) #generate noise per the book's definition (sigma_noise = 0.2)
tn = tn_synthetic + noise          #add noise to syntheticc data

##########################################
#take the targets one by one to determine mus and covariances of the w's (estimates for a0 & a1)
#define the prior mus and covariances assuming 0 mean unit variance
#no need because the update equations take this into account
#mu_0 = 0
#cov_0 = (alpha) * np.identity(2)

##########################################
#update the prior for each new observation
#muN = mu_0
#SN = cov_0
plot_vals = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 25, 50, N-1]
x, y = np.mgrid[-1:1:.01, -1:1:.01]
pos = np.dstack((x, y))

iota = np.array([[1 , xn[0]]])       #basis functions are 1 and x
for i in range(N):
    iotanew = np.array([[1, xn[i]]])
    if not np.array_equal(iota, iotanew):
        iota = np.concatenate((iota, iotanew), axis=0)      #add row to iota
    iotaT = iota.T
    SN_inv = (alpha * np.identity(2)) + (beta_noise * iotaT.dot(iota))
    SN = np.linalg.inv(SN_inv)
    tn_N = tn[0:i+1].T
    muN = beta_noise * SN.dot(iotaT).dot(tn_N)
    if i in plot_vals:
        rv = multivariate_normal(muN, SN)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.set_title('Contour plot of weight estimates at observation ' + str(i + 1), fontweight='bold')
        ax2.contourf(x, y, rv.pdf(pos))


