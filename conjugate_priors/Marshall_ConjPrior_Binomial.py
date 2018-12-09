#Jessica Marshall
#ECE414 Machine Learning
#Conjugate Priors Programming Assignment
#binomial

##########################################
#import libraries
import math
import numpy as np
from scipy.stats import binom, beta
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
#the conjugate prior of the binomial is a beta
#define  parameters of initial prior
a_0 = 3
b_0 = 20

#plot initial prior and likelihood
fig1 = plt.figure()
ax11 = fig1.add_subplot(1, 1, 1)
ax12 = fig1.add_subplot(1, 1, 1)
x = np.linspace(0, 1, N)
y2 = beta.pdf(x, a_0, b_0)       #plot initial prior
prior = ax12.plot(x, y2, 'g--', label='Original Prior')
handles1, labels1 = ax11.get_legend_handles_labels()
ax11.legend(handles1, labels1)
ax11.set_title('Conjugate Prior Update - Binomial with p = ' + str(p), fontweight='bold')

##########################################
#update equations for Gaussian of known variance, unknown mean
#initial hyperparameters
update_a = a_0
update_b = b_0
plot_list = [0, 1, 2, 3, 4, 9, 19, 49, N-1]

for j in range(0, N):       #N is the obseration in question, one index off
    n_update = j + 1    
    sum_xn = sum(X[0:n_update])
    sum_Ni = n*(j+1)
    
    update_a = update_a + sum_xn      #where sum is the sum of observations up to xn
    update_b = update_b + sum_Ni - sum_xn

    
    if j in plot_list:
        fig2 = plt.figure()
        ax21 = fig2.add_subplot(1, 1, 1)
        y3 = beta.pdf(x, update_a, update_b)
        ax21.plot(x, y2, 'g--', label='Original Prior')
        ax21.plot(x, y3, 'g', label='Updated Prior on Observation ' + str(j+1))
        handles2, labels2 = ax21.get_legend_handles_labels()
        ax21.legend(handles2, labels2)
        ax21.set_title('Conjugate Prior Update - Binomial with p = ' + str(p), fontweight='bold')
