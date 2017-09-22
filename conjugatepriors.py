#Jessica Marshall
#ECE-302
#Programming Assignment #3

import math
import numpy as np
import matplotlib.pyplot as plt

# Scenario 1

#define constants
N = 300              #number of iterations excluding bias
h = 0.5             #fixed

# implement MLE/MMSE
mu_theta = 1       #change this
sigma_theta = .5     #change this
var_theta = sigma_theta*sigma_theta

mu_v = 0            #fixed
sigma_v = 1        #change this
var_v = sigma_v*sigma_v

mu_x = h*mu_theta
var_x = (h*h*var_theta) + var_v
sigma_x = math.sqrt(var_x)

X = h*np.random.normal(mu_x, sigma_x, N) + np.random.normal(mu_v, sigma_v, N)      #vector of X observations
X_alt1 = np.random.normal(mu_x, sigma_x, N) + np.random.normal(mu_v, sigma_v, N)
X_alt2 = np.random.uniform(-1, 3, N)


#RUN MAX LIKELIHOOD

ML = np.zeros(N)
ML_alt1 = np.zeros(N)
ML_alt2 = np.zeros(N)

for i in range(1, N+1):
    ML[i-1] = (1/(i*h))*(X[0:i].sum())
    ML_alt1[i-1] = (1/(i*h))*(X_alt1[0:i].sum())
    ML_alt2[i-1] = (1/(i*h))*(X_alt2[0:i].sum())

## PLOT MMSE OUTPUT
    
line1 = plt.plot(np.linspace(1, N-1, N-1), ML[1:N+1])
line2 = plt.plot(np.linspace(1, N-1, N-1), ML_alt1[1:N+1])
line3 = plt.plot(np.linspace(1, N-1, N-1), ML_alt2[1:N+1])
plt.title(r'$\mathrm{Convergence\ of\ ML}$')
plt.xlabel('number of iterations')
plt.ylabel('estimator')
plt.show()

print('ML estimator results')
print('mu_x: ', mu_x)
print('mu_x_hat: ', ML[N-1])  
print('mu_x_hat_alt1: ', ML_alt1[N-1])   
print('mu_x_hat_alt2: ', ML_alt2[N-1])   
