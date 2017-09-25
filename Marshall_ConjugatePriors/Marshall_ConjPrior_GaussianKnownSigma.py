#Jessica Marshall
#ECE414 Machine Learning
#Conjugate Priors Programming Assignment
#Gaussian sigma

##########################################
#import libraries
import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt

##########################################
#generate normally distributed observations with awgn
#here we assume the variance is known, the mean is unknown parameter
#this is the likelihood function in Baye's rule
mu = 0          #known
sigma = 2
variance = sigma**2
precision = 1/variance      #unknown -> trying to estimate
N = 100        #number of observations
X_normal = np.random.normal(mu, sigma, N)      #data without noise

mu_noise = 0
sigma_noise = sigma/2
variance_noise = sigma_noise**2
X_awgn = np.random.normal(mu_noise, sigma_noise, N)     #noise

X = X_normal + X_awgn           #a vector of observations as an example/for plotting

##########################################
#the conjugate prior of the Gaussian with known mean is Gamma (hyperparameters estimate precision)
#define  parameters of initial prior
a = 10
b = .01

#plot initial prior and likelihood
fig1 = plt.figure()
ax12 = fig1.add_subplot(1, 1, 1)
x = np.linspace(precision - 1, precision + 1, N)
y2 = gamma.pdf(x, a=a, scale=1/b)       #plot initial prior, scale is theta = 1/beta
prior = ax12.plot(x, y2, 'g--', label='Original Prior')
handles1, labels1 = ax12.get_legend_handles_labels()
ax12.legend(handles1, labels1)
ax12.set_title('Conjugate Prior Update - Gaussian with known mean, unknown precision = ' + str(precision), fontweight='bold')

##########################################
#update equations for Gaussian of known variance, unknown mean
#initial hyperparameters
update_a = a
update_b = b
plot_list = [0, 1, 2, 3, 4, 9, 19, 49, N-1]

for j in range(0, N):       #N is the obseration in question, one index off
    n_update = j + 1    
    sum_xn_mu_squared = sum((X[0:n_update] - mu)**2)
    
    update_a = update_a + (n_update/2)
    update_b = update_b + (sum_xn_mu_squared/2)

    if j in plot_list:
        fig2 = plt.figure()
        ax21 = fig2.add_subplot(1, 1, 1)
        y3 = gamma.pdf(x, a=update_a, scale=1/update_b)
        ax21.plot(x, y2, 'g--', label='Original Prior')
        ax21.plot(x, y3, 'g', label='Updated Prior on Observation ' + str(j+1))
        handles2, labels2 = ax21.get_legend_handles_labels()
        ax21.legend(handles2, labels2)
        ax21.set_title('Conjugate Prior Update - Gaussian with known mean, unknown precision = ' + str(precision), fontweight='bold')

            
    




