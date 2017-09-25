#Jessica Marshall
#ECE414 Machine Learning
#Conjugate Priors Programming Assignment
#MSE plots - Gaussian mu

##########################################
#import libraries

import math
import numpy as np
import matplotlib.pyplot as plt

##########################################
#generate normally distributed observations with awgn
#here we assume the variance is known, the mean is unknown parameter
#this is the likelihood function in Baye's rule
mu = 0
sigma = 10
variance = sigma**2
precision = 1/variance
N = 150        #number of observations
mu_noise = 0
sigma_noise = sigma/2
variance_noise = sigma_noise**2

##########################################
#mean squared error of maximum likelihood 
numIter = 10                   #times we run the estimator (requires new data)
ML = np.zeros((numIter, N))    #hold max likelihood values of each observation update for each estimator run
data = np.zeros((numIter, N))

for i in range(0, numIter):                                 #using the new mu and sigma, run estimator multiple times by generating list of observations multiple times
    X_ML = np.random.normal(mu+ mu_noise, math.sqrt(variance + variance_noise), N)     #generate 1000 observations
    data[i] = X_ML
    
    for j in range(0, N):
        ML[i, j] = (1/(j+1))*(X_ML[:j+1].sum())                  #store ML estimate for this observation index
            
#for each observation "set" calculate the MSE of each ML estimate
SE = (((ML - mu)**2))
MSE_ML = np.mean(SE, axis=0)

#plot mean squared error of max likelihood estimate at each observation
fig2 = plt.figure()
x = np.linspace(1, N, N)
ax21 = fig2.add_subplot(1, 1, 1)
ax21.plot(x, MSE_ML, 'b', label='MSE of Max Likelihood Estimate')
ax21.set_title('MSE of Max Likelihood Estimate and Conjugate Prior - Mean of Gaussian with Known Variance', fontweight='bold')

##########################################
#update equations for Gaussian

#the conjugate prior of the Gaussian with known variance is a Gaussian
#define hyperparameters of initial prior
mu_0 = [-5, 2, 6]             #choose 3 different hyperparameter mus & sigmas
sigma_0 = [10, 20, 2]                    #make this very broad
precision_0 = np.ones(len(sigma_0))/(np.power(sigma_0, 2))
color = ['y','r', 'c']

SE_conjprior= np.zeros((numIter, N))

for l in range(0, len(mu_0)):
    #do this for multiple different hyperparameters
    update_mu = mu_0[l]
    update_sigma = sigma_0[l]
    
    for i in range(0, numIter):
        X_ML = data[i]     #use same observations as max likelihood for each trial to ensure comparability
        
        for j in range(0, N):       #N is the obseration in question, one index off
            n_update = j + 1    
            sum_xn = sum(X_ML[0:n_update])
        
            update_mu = ((mu_0[l]*precision_0[l]) + (sum_xn*precision))/((precision_0[l])+(n_update/variance))   #where sum is the sum of observations up to xn
            update_variance = 1/(precision_0[l] + n_update*precision)
            update_sigma = math.sqrt(update_variance)
        
            #mean squared error of update parameters
            SE_conjprior[i, j] = (1/(j+1))*((mu-update_mu)**2)
       
    #plot MSE of conjugate prior update at each obsercation
    MSE_conjprior = np.mean(SE_conjprior, axis=0)
    ax22 = fig2.add_subplot(1, 1, 1)
    ax22.plot(x, MSE_conjprior, color[l], label='MSE of Conjugate Prior: mu = ' + str(mu_0[l]) + ', sigma = ' + str(sigma_0[l]))
    handles, labels = ax21.get_legend_handles_labels()
    ax22.legend(handles, labels)    
    ax22.set_xlabel('Observations')
    ax22.set_ylabel('Mean Squared Error')

    




