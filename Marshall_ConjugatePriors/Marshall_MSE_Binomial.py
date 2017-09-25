#Jessica Marshall
#ECE414 Machine Learning
#Conjugate Priors Programming Assignment
#MSE plots - binomial

##########################################
#import libraries

import numpy as np
import matplotlib.pyplot as plt

##########################################
#generate binomially distributed observations with awgn
#here we assume the probability p is the unknown parameter
#this is the likelihood function in Baye's rule
p = 0.7
N = 100        #number of observations
n = 10
X_binom = np.random.binomial(n, p, size=N)      #data without noise

mu_noise = 0
sigma_noise = 1
X_awgn = np.random.normal(mu_noise, sigma_noise, N)     #noise

X = X_binom + X_awgn           #a vector of observations

##########################################
#mean squared error of maximum likelihood 
numIter = 10                   #times we run the estimator (requires new data)
ML = np.zeros((numIter, N))    #hold max likelihood values of each observation update for each estimator run
data = np.zeros((numIter, N))

for i in range(0, numIter):                                 #using the new mu and sigma, run estimator multiple times by generating list of observations multiple times
    X_ML = np.random.binomial(n, p, size=N) + np.random.normal(mu_noise, sigma_noise, N)      #generate 1000 observations
    data[i] = X_ML
    
    for j in range(0, N):
        ML[i, j] = sum(X_ML[:j+1])/(n*(j+1))                 #store ML estimate for this observation index


#for each observation "set" calculate the MSE of each ML estimate
SE = (((ML - p)**2))
MSE_ML = np.mean(SE, axis=0)

#plot mean squared error of max likelihood estimate at each observation
fig2 = plt.figure()
x = np.linspace(1, N, N)
ax21 = fig2.add_subplot(1, 1, 1)
ax21.plot(x, MSE_ML, 'b', label='MSE of Max Likelihood Estimate')
ax21.set_title('MSE of Max Likelihood Estimate and Conjugate Prior - Binomial', fontweight='bold')

##########################################
#update equations for binomial

#the conjugate prior of the binomial is a beta
#define hyperparameters of initial prior
a_0 = [1, 3, 1]             #choose 3 different hyperparameter mus & sigmas
b_0 = [3, 4, 1]                    #make this very broad
color = ['y','r', 'c']

SE_conjprior= np.zeros((numIter, N))

for l in range(0, len(a_0)):
    #do this for multiple different hyperparameters
    update_a = a_0[l]
    update_b = b_0[l]
    
    for i in range(0, numIter):
        X_ML = data[i]     #use same observations as max likelihood for each trial to ensure comparability
        
        for j in range(0, N):       #N is the obseration in question, one index off
            n_update = j + 1    
            sum_xn = sum(X_ML[0:n_update])
            sum_Ni = n*(j+1)
    
            update_a = update_a + sum_xn
            update_b = update_b + sum_Ni - sum_xn
        
            p_est = update_a/(update_a + update_b)
            #mean squared error of update parameters
            SE_conjprior[i, j] = (1/(j+1))*((p-p_est)**2)
       
    #plot MSE of conjugate prior update at each obsercation
    MSE_conjprior = np.mean(SE_conjprior, axis=0)
    ax22 = fig2.add_subplot(1, 1, 1)
    ax22.plot(x, MSE_conjprior, color[l], label='MSE of Conjugate Prior - Probability of Binomial: a = ' + str(a_0[l]) + ', b = ' + str(b_0[l]))
    handles, labels = ax21.get_legend_handles_labels()
    ax22.legend(handles, labels)    
    ax22.set_xlabel('Observations')
    ax22.set_ylabel('Mean Squared Error')

    




