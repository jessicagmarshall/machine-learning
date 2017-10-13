#Jessica Marshall
#ECE414 Machine Learning
#Binary Classification
#equations 4.73, 4.75, 4.78, 4.80

##########################################
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import math

##########################################
#define functions
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

##########################################
#constants
N = 100                          #number of training datapoints
Ntest = 50                       #number of testing datapoints
pi0 = .7                         #probability of class 0
N0 = int(N * pi0)                #number of datapoints from class 0
N1 = int(N * (1 - pi0))

N0test = int(Ntest * pi0)                #number of datapoints from class 0
N1test = int(Ntest * (1 - pi0))

mu0 = 1
mu1 = -1
sigma = 1                           #covariance is [1, 0; 0, 1]

x0 = np.random.normal(mu0, sigma, (N0 + N0test, 2))      #each datapoint is a row
ones = np.ones((N0 + N0test, 1))                         #class 0 has target 1
x0 = np.append(x0, ones, axis=1)                #targets in last column

x1 = np.random.normal(mu1, sigma, (N1 + N1test, 2))
zeros = np.zeros((N1 + N1test, 1))                       #class 1 has target 0
x1 = np.append(x1, zeros, axis=1)               #targets in last column

x2 = np.concatenate([x0, x1],axis=0)            #smush data together
xnall = np.random.permutation(x2)                  #shuffle order of datapoints

xn = xnall[0:N, :]
xntest = xnall[N:, :]                   #test data

##########################################
#use equation 4.73 to predict pi0
pi0_estimate = np.zeros(N)

for i in range(0, N):
    pi0_estimate[i] = (1/(i+1)) * sum(xn[0:i+1, 2])
    
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_xlabel('observations')
ax1.set_ylabel('pi0 estimate')
ax1.set_title('Max Likelihood Estimate of pi0, truth is 0.7', fontweight='bold')
x = np.linspace(0, 1, N)
ax1.plot(x, pi0_estimate)


##########################################
#use equation 4.75 to estimate mu0
N0 = sum(xn[:, 2])

x0_temp = np.array([np.multiply(xn[:, 0], xn[:, 2])]).T
x1_temp = np.array([np.multiply(xn[:, 1], xn[:, 2])]).T
x2_temp = np.concatenate([x0_temp, x1_temp],axis=1)

mu0_estimate = (1/N0)*np.sum(x2_temp, axis=0)
print('mu0 estimate =', mu0_estimate)
print('mu0 ground truth =', [mu0, mu0], '\n')

##########################################
#use equation 4.75 to estimate mu1
N1 = sum((1 - xn[:, 2]))

x0_temp = np.array([np.multiply(xn[:, 0], (1 - xn[:, 2]))]).T
x1_temp = np.array([np.multiply(xn[:, 1], (1 - xn[:, 2]))]).T
x2_temp = np.concatenate([x0_temp, x1_temp],axis=1)

mu1_estimate = (1/N1)*np.sum(x2_temp, axis=0)
print('mu1 estimate =', mu1_estimate)
print('mu1 ground truth =', [mu1, mu1], '\n')

##########################################
#use equation 4.78 - 4.80 to estimate S
S0_temp = 0
S1_temp = 0

for i in range(0, N):
    if(xn[i, 2] == 1):       #if observation is in class 0
        temp = np.array([xn[i, :2] - mu0_estimate]).T
        S0_temp += temp.dot(temp.T)
    if(xn[i, 2] == 0):       #if observation is in class 1
        temp = np.array([xn[i, :2] - mu1_estimate]).T
        S1_temp += temp.dot(temp.T)

S0 = (1/N0) * S0_temp
S1 = (1/N1) * S1_temp
S = (N0/N) * S0 + (N1/N) * S1
print('S estimate =', S)
print('S truth =', np.identity(2), '\n')

##########################################
#test this on some test data!
PC0 = np.zeros(Ntest)
prediction = np.zeros(Ntest)
truth = xntest[:, 2]

w = np.linalg.inv(S).dot(mu0_estimate - mu1_estimate)
w0 = (-0.5 * mu0_estimate.dot(np.linalg.inv(S).dot(mu0_estimate))) + (0.5 * mu1_estimate.dot(np.linalg.inv(S).dot(mu1_estimate))) + (np.log(pi0/(1-pi0))) 

for i in range(Ntest):
    PC0[i] =  sigmoid(w.dot(xntest[i, :2]) + w0)
    if PC0[i] > 0.5:
        prediction[i] = 1

error = 1 - (np.sum(truth == prediction)/Ntest)
print('test error = ', 100 * error, '%')
