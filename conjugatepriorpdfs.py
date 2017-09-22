#Jessica Marshall
#ECE414 Machine Learning
#Conjugate Priors Programming Assignment

import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

#generate normally distributed observations with awgn
#here we assume the variance is known, the mean is unknown parameter
#this is the likelihood function in Baye's rule
mu = 1
sigma = 10
variance = sigma**2
precision = 1/variance
N = 10        #number of observations
X_normal = np.random.normal(mu, sigma, N)      #data without noise

mu_noise = 0
sigma_noise = sigma/100
variance_noise = sigma_noise**2
X_awgn = np.random.normal(mu_noise, sigma_noise, N)     #noise

X = X_normal + X_awgn           #vector of observations

#the conjugate prior of the Gaussian with known variance is a Gaussian
#define  parameters of initial prior
mu_0 = 1
sigma_0 = 10        #make this very broad
precision_0 = 1/(sigma_0**2)

#plot initial prior and likelihood
#fig1 = plt.figure()
#ax1 = fig1.add_subplot(1, 1, 1)
#ax2 = fig1.add_subplot(1, 1, 1)
#x = np.linspace(mu_0 - 10*sigma_0, mu_0 + 10*sigma_0, N)
#y1 = norm.pdf(x, loc=(mu + mu_noise), scale=(math.sqrt(variance + variance_noise)))     #loc is mu, scale is std dev
#y2 = norm.pdf(x, loc=mu_0, scale=sigma_0)       #plot initial prior
#
#ax1.plot(x, y1, 'r')
#ax2.plot(x, y2, 'g')

#update equations for Gaussian
paramList = np.zeros((2, N))   #first row is mu, second row is sigma, use these to plot later
numIter = 15                   #times we run the estimator (requires new data)
ML = np.zeros((N, numIter))    #hold max likelihood values of each observation update for each estimator run
MSE = np.zeros(N)              #hold mean squared error of average max likelihood

for i in range(0, N):       #N is the obseration in question, one index off
    n_update = i + 1    
    sum_xn = sum(X[0:n_update])
    
    update_mu = ((mu_0*precision_0) + (sum_xn*precision))/((precision_0)+(n_update/variance))   #where sum is the sum of observations up to xn
    update_variance = 1/(precision_0 + n_update*precision)
    update_sigma = math.sqrt(update_variance)
    
    #paramList[0, i] = update_mu
    #paramList[1, i] = update_sigma
    
#    #plot new observations
#    fig = plt.figure()
#    ax3 = fig.add_subplot(1, 1, 1)
#    y3 = norm.pdf(x, loc=update_mu, scale=update_sigma)
#    ax3.plot(x, y1, 'r',  x, y2, 'g--', x, y3, 'g')

    #maximum likelihood estimation for Gaussian mean is the sample mean
    for j in range(0, numIter):         #using the new mu and sigma, generate list of observations multiple times
        X_ML = np.random.normal(update_mu, update_sigma, N)     #regenerate 1000 observations
        ML[i, j] = (1/(i+1))*(X_ML[0:j].sum())                  #store ML estimate for this observation index
    
    #for each observation "set" calculate the MSE of the ML estimate
    mu_vector = np.ones(numIter)*mu
    MSE[i] = (1/n_update)*(((ML[i] - mu_vector)**2).sum())

#plot max likelihood estimate of multiple 
x = np.linspace(0, N-1, N)
plt.plot(x, MSE)





