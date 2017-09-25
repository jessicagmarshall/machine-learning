#Jessica Marshall
#ECE414 Machine Learning
#Conjugate Priors Programming Assignment
#MSE plots - Gaussian sigma

##########################################
#import libraries

import math
import numpy as np
import matplotlib.pyplot as plt

##########################################
#generate normally distributed observations with awgn
#here we assume the mean is known, the standard deviation is unknown parameter
#this is the likelihood function in Baye's rule
mu = 0                  #known
sigma = 2
variance = sigma**2
precision = 1/variance   #unknown -> trying to estimate
N = 100        #number of observations
mu_noise = 0
sigma_noise = sigma/2
variance_noise = sigma_noise**2
X = np.random.normal(mu + mu_noise, (math.sqrt(variance + variance_noise)), N)      #mus and variances add

##########################################
#mean squared error of maximum likelihood 
numIter = 10                   #times we run the estimator (requires new data)
ML = np.zeros((numIter, N))    #hold max likelihood values of each observation update for each estimator run
data = np.zeros((numIter, N))

for i in range(0, numIter):                                 #using the new mu and sigma, run estimator multiple times by generating list of observations multiple times
    X_ML = np.random.normal(mu+ mu_noise, math.sqrt(variance + variance_noise), N)     #generate 1000 observations
    data[i] = X_ML
    
    for j in range(0, N):
        ML[i, j] = (1/(j+1))*(((X_ML[:j+1]- mu)**2).sum())                 #store ML estimate of variance for this observation index
            
#for each observation "set" calculate the MSE of each ML estimate
SE = ((1/ML) - precision)**2
MSE_ML = np.mean(SE, axis=0)

#plot mean squared error of max likelihood estimate at each observation
fig2 = plt.figure()
x = np.linspace(1, N, N)
ax21 = fig2.add_subplot(1, 1, 1)
ax21.plot(x, MSE_ML, 'b', label='MSE of Max Likelihood Estimate')
ax21.set_title('MSE of Max Likelihood Estimate and Conjugate Prior - Precision of Gaussian with Known Mean', fontweight='bold')

##########################################
#update equations for Gaussian

#the conjugate prior of the Gaussian with known mean is a Gamma
#define hyperparameters of initial prior
a_0= [5, 2, 10]             #choose 3 different hyperparameter a's and b's
b_0 = [4, 2, 5]                  #make this very broad
color = ['y','r', 'c']

SE_conjprior= np.zeros((numIter, N))

for l in range(0, len(a_0)):
    #do this for multiple different hyperparameters
    update_a = a_0[l]
    update_b = b_0[l]
    
    for i in range(0, numIter):
        X_ML = data[i]     #use same observations as max likelihood for each trial to ensure comparability
        
        for j in range(0, N):       #N is the observation in question, one index off
            n_update = j + 1    
            sum_xn_mu_squared = sum((X_ML[0:n_update] - mu)**2)
        
            update_a = update_a + (n_update/2)
            update_b = update_b + (sum_xn_mu_squared/2)
        
            #mean squared error of precision
            precision_est = update_a/update_b 
            SE_conjprior[i, j] = (1/(j+1))*((precision_est-precision)**2)
       
    #plot MSE of conjugate prior update at each obsercation
    MSE_conjprior = np.mean(SE_conjprior, axis=0)
    ax22 = fig2.add_subplot(1, 1, 1)
    ax22.plot(x, MSE_conjprior, color[l], label='MSE of Conjugate Prior: a = ' + str(a_0[l]) + ', b = ' + str(b_0[l]))
    handles, labels = ax21.get_legend_handles_labels()
    ax22.legend(handles, labels)    
    ax22.set_xlabel('Observations')
    ax22.set_ylabel('Mean Squared Error')