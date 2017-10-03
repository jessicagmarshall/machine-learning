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
N = [1, 2, 4, 25]
beta_noise = 25                     #defined by the book
alpha = 2.0                         #defined by the book

#plot ground truth
xtruth = np.linspace(0, 1, 100)
ytruth = np.sin(2*np.pi*xtruth)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(xtruth, ytruth, 'g')

#generate data
xn = np.random.uniform(0, 1, N[3])
t_synthetic = np.sin(2*np.pi*xn)
noise = np.random.normal(0, 0.15, N[3])
tn = t_synthetic + noise
#plt.scatter(xn, yn)

mu = np.linspace(0, 1, 9)       #define mus of radial Gaussian basis functions
s = 0.1                         #define s in each Gaussian basis function as 0.1 for now

#computing Mn and Sn for N = 1
#unit mean, alpha * identity covariance assumption
#iota where N = 0 is a 1x9 matrix (1 observation by 9 basis functions)

x_temp = np.ones(9) * xn[0]
iota = np.array([np.exp(-np.multiply(x_temp - mu, x_temp - mu)/(2*s*s))])
iotaT = iota.T
SN_inv = alpha*np.identity(9) + beta_noise * iotaT.dot(iota)
SN = SN_inv.T
mN = beta_noise * SN.dot(iotaT).dot(tn[1])


muN = np.zeros(100)
for i in range(len(xtruth)):
    x_temp = np.ones(9) * xtruth[i]
    phi = np.array([np.exp(-np.multiply(x_temp - mu, x_temp - mu)/(2*s*s))])
    muN[i] = np.dot(mN.T.flatten(), phi.flatten())

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(xtruth, muN)


