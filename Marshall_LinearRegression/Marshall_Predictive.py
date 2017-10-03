#Jessica Marshall
#ECE414 Machine Learning
#Linear Regression Assignment
#recreate figure 3.8

##########################################
#import libraries
import numpy as np
import matplotlib.pyplot as plt

##########################################
#constants
N = [0, 1, 3, 24, 99]
beta_noise = 25                     #defined by the book
alpha = 2.0                         #defined by the book

#plot ground truth
xtruth = np.linspace(0, 1, 100)
ytruth = np.sin(2*np.pi*xtruth)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_title('Ground Truth', fontweight='bold')
ax1.plot(xtruth, ytruth, 'g')

#generate data
xn = np.random.uniform(0, 1, max(N) + 1)
t_synthetic = np.sin(2*np.pi*xn)
noise = np.random.normal(0, 0.15, max(N) + 1)
tn = t_synthetic + noise
#plt.scatter(xn, yn)

mu = np.linspace(0, 1, 9)       #define mus of radial Gaussian basis functions
s = 0.1                         #define s in each Gaussian basis function as 0.1 for now

#computing Mn and Sn for N = 1
#unit mean, alpha * identity covariance assumption
#iota where N = 0 is a 1x9 matrix (1 observation by 9 basis functions)


x_temp = np.ones(9) * xn[0]
iota = np.array([np.exp(-np.multiply(x_temp - mu, x_temp - mu)/(2*s*s))])      #basis functions are 9 Gaussians
for i in range(max(N) + 1):
    x_temp = np.ones(9) * xn[i]
    iotanew = np.array([np.exp(-np.multiply(x_temp - mu, x_temp - mu)/(2*s*s))])
    if not np.array_equal(iota, iotanew):
        iota = np.concatenate((iota, iotanew), axis=0)      #add row to iota
    iotaT = iota.T
    SN_inv = (alpha * np.identity(9)) + (beta_noise * iotaT.dot(iota))
    SN = np.linalg.inv(SN_inv)
    tn_N = tn[0:i+1].T
    mN = beta_noise * SN.dot(iotaT).dot(tn_N)
    
    #plot estimate
    muN = np.zeros(100)
    SN_x = np.zeros(100)
    for j in range(len(xtruth)):
        x_temp = np.ones(9) * xtruth[j]
        phi = np.array([np.exp(-np.multiply(x_temp - mu, x_temp - mu)/(2*s*s))])
        muN[j] = np.dot(mN.T.flatten(), phi.flatten())
        SN_x[j] = (1/beta_noise) + np.dot(np.dot(phi, SN), phi.T)
    if i in N:
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.set_title('Estimate on observation ' + str(i+1), fontweight='bold')
        ax2.plot(xtruth, muN)
        ax2.plot(xtruth, ytruth, 'g')
        ax2.plot(xn[0:i+1], tn[0:i+1], 'r+')
        ax2.fill_between(xtruth, muN + SN_x, muN - SN_x, facecolor='red', alpha=.125)
        ax2.plot(xtruth, muN + SN_x, 'r', alpha=.5)
        ax2.plot(xtruth, muN - SN_x, 'r', alpha=.5)




