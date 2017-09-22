#Jessica Marshall
#ECE414 Machine Learning
#Conjugate Priors Programming Assignment
#MSE plots

import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

#generate normally distributed observations with awgn
#here we assume the variance is known, the mean is unknown parameter
#this is the likelihood function in Baye's rule
mu = 0
sigma = 10
variance = sigma**2
precision = 1/variance
N = 100        #number of observations
mu_noise = 0
sigma_noise = sigma/2
variance_noise = sigma_noise**2
X = np.random.normal(mu + mu_noise, (math.sqrt(variance + variance_noise)), N)

#the conjugate prior of the Gaussian with known variance is a Gaussian
#define  parameters of initial prior
mu_0 = 50
sigma_0 = 10        #make this very broad
precision_0 = 1/(sigma_0**2)

#plot initial prior and likelihood
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax2 = fig1.add_subplot(1, 1, 1)
x = np.linspace(mu_0 - 10*sigma_0, mu_0 + 10*sigma_0, N)
y1 = norm.pdf(x, loc=(mu + mu_noise), scale=(math.sqrt(variance + variance_noise)))     #loc is mu, scale is std dev
y2 = norm.pdf(x, loc=mu_0, scale=sigma_0)       #plot initial prior
likelihood = ax1.plot(x, y1, 'r', label='Likelihood')
prior = ax2.plot(x, y2, 'g', label='Prior')
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels)

#plt.legend(handles=[y1, y2])

#mean squared error of maximum likelihood 
numIter = 100                   #times we run the estimator (requires new data)
ML = np.zeros((numIter, N))    #hold max likelihood values of each observation update for each estimator run
multiplier = np.zeros((numIter, N))     #hold the 1/n to calculate MSE

for i in range(0, numIter):                                 #using the new mu and sigma, run estimator multiple times by generating list of observations multiple times
    X_ML = np.random.normal(mu+ mu_noise, math.sqrt(variance + variance_noise), N)     #generate 1000 observations
    for j in range(0, N):
        ML[i, j] = (1/(j+1))*(X_ML[:i+1].sum())                  #store ML estimate for this observation index
        multiplier[i, j] = 1/(j+1)
            
    #for each observation "set" calculate the MSE of each ML estimate
SE = multiplier*(((ML - mu)**2))
MSE_ML = np.mean(SE, axis=0)

#plot mean squared error of max likelihood estimate at each observation
fig2 = plt.figure()
x = np.linspace(1, N, N)
ax1 = fig2.add_subplot(1, 1, 1)
ax1.plot(x, MSE_ML)

#update equations for Gaussian
paramList = np.zeros((2, N))   #first row is mu, second row is sigma, use these to plot later
SE_conjprior= np.zeros((numIter, N))

#initial parameters
update_mu = mu_0
update_sigma = sigma_0

for i in range(0, numIter):
    X_ML = np.random.normal(update_mu, update_sigma, N)     #generate new observations
    
    for j in range(0, N):       #N is the obseration in question, one index off
        n_update = j + 1    
        sum_xn = sum(X[0:n_update])
    
        update_mu = ((mu_0*precision_0) + (sum_xn*precision))/((precision_0)+(n_update/variance))   #where sum is the sum of observations up to xn
        update_variance = 1/(precision_0 + n_update*precision)
        update_sigma = math.sqrt(update_variance)
    
        #paramList[0, i] = update_mu
        #paramList[1, i] = update_sigma
    
#        #plot new observations
#        fig = plt.figure()
#        ax3 = fig.add_subplot(1, 1, 1)
#        y3 = norm.pdf(x, loc=update_mu, scale=update_sigma)
#        ax3.plot(x, y1, 'r',  x, y2, 'g--', x, y3, 'g')

        #mean squared error of update parameters
        SE_conjprior[i, j] = (1/(j+1))*((mu-update_mu)**2)
   
MSE_conjprior = np.mean(SE_conjprior, axis=0)
plt.plot(x, MSE_conjprior, 'r')
            
    




