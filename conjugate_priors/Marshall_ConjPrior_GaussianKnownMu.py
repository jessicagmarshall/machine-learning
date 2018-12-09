#Jessica Marshall
#ECE414 Machine Learning
#Conjugate Priors Programming Assignment
#Gaussian mu

##########################################
#import libraries
import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

##########################################
#generate normally distributed observations with awgn
#here we assume the variance is known, the mean is unknown parameter
#this is the likelihood function in Baye's rule
mu = 0
sigma = 10
variance = sigma**2
precision = 1/variance
N = 100        #number of observations
X_normal = np.random.normal(mu, sigma, N)      #data without noise

mu_noise = 0
sigma_noise = sigma/2
variance_noise = sigma_noise**2
X_awgn = np.random.normal(mu_noise, sigma_noise, N)     #noise

X = X_normal + X_awgn           #a vector of observations as an example/for plotting

##########################################
#the conjugate prior of the Gaussian with known variance is a Gaussian
#define  parameters of initial prior
mu_0 = 50
sigma_0 = 10        #make this very broad
precision_0 = 1/(sigma_0**2)

#plot initial prior and likelihood
fig1 = plt.figure()
ax11 = fig1.add_subplot(1, 1, 1)
ax12 = fig1.add_subplot(1, 1, 1)
x = np.linspace(mu_0 - 10*sigma_0, mu_0 + 10*sigma_0, N)
y1 = norm.pdf(x, loc=(mu + mu_noise), scale=(math.sqrt(variance + variance_noise)))     #loc is mu, scale is std dev
y2 = norm.pdf(x, loc=mu_0, scale=sigma_0)       #plot initial prior
likelihood = ax11.plot(x, y1, 'r', label='Likelihood')
prior = ax12.plot(x, y2, 'g', label='Original Prior')
handles1, labels1 = ax11.get_legend_handles_labels()
ax11.legend(handles1, labels1)
ax11.set_title('Conjugate Prior Update - Gaussian with known sigma, unknown mean = ' + str(mu), fontweight='bold')

##########################################
#update equations for Gaussian of known variance, unknown mean
#initial hyperparameters
update_mu = mu_0
update_sigma = sigma_0
plot_list = [0, 1, 2, 3, 4, 9, 19, 49, N-1]

for j in range(0, N):       #N is the obseration in question, one index off
    n_update = j + 1    
    sum_xn = sum(X[0:n_update])
    
    update_mu = ((mu_0*precision_0) + (sum_xn*precision))/((precision_0)+(n_update/variance))   #where sum is the sum of observations up to xn
    update_variance = 1/(precision_0 + n_update*precision)
    update_sigma = math.sqrt(update_variance)

    
    if j in plot_list:
        fig2 = plt.figure()
        ax21 = fig2.add_subplot(1, 1, 1)
        y3 = norm.pdf(x, loc=update_mu, scale=update_sigma)
        ax21.plot(x, y1, 'r', label='Likelihood')
        ax21.plot(x, y2, 'g--', label='Original Prior')
        ax21.plot(x, y3, 'g', label='Updated Prior on Observation ' + str(j+1))
        handles2, labels2 = ax21.get_legend_handles_labels()
        ax21.legend(handles2, labels2)
        ax21.set_title('Conjugate Prior Update - Gaussian with known sigma, unknown mean = ' + str(mu), fontweight='bold')

            
    




