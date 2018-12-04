import numpy as np

#1.1.
#Find the loss at the starting point and after two iterations, using two different
#learning rates: 1 and 0.1.
#1.2.
#For each learning rate, explain why did the gradient descent succeed/fail?
print('\n\n~~~~~~~~~~~~~~ Gradient descent ~~~~~~~~~~~~~~')
Y = np.array([1,3,7]).reshape(1,3)
print(Y)
X = np.array([[1,1,1],[0,1,2],[0,1,4]])
Theta = np.array([2,2,0]).reshape(1,3)
Alpha = 0.1 # Learning rate

print('Old Theta: ',Theta)

Max_iterrasion = 100 ; It_index = 1;
while(It_index <= Max_iterrasion):
   dl_to_dTheta = np.dot(np.dot(Theta,X) - Y ,X) / len(X)
   New_theta = Theta - Alpha*dl_to_dTheta
   Hypothesis= New_theta[0,0] + New_theta[0,1]*X[1,:] + New_theta[0,2]*(X[1,:]**2) ; Hypothesis = Hypothesis.reshape(3,1)
   #Loss = (1/(2*len(X))) * np.sum((Hypothesis-Y.T)**2)
   Loss = 0.5 * np.sum((Hypothesis-Y.T)**2)
   print('Loss for iteration: ',It_index, 'is: ',Loss)
   print('test:', Hypothesis)
   Theta = New_theta
   It_index+=1

print('New theta: ',New_theta)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#1.3.
#Repeat the process using LR=0.1, but this time with momentum γ = 0 .9 .
print('\n\n~~~~~~~~~~~~~~ Momentum ~~~~~~~~~~~~~~')
print('Old Theta: ',Theta)
Theta = np.array([2,2,0]).reshape(1,3)

Max_iterrasion = 100 ; It_index = 1;
Gamma = 0.9  ;  LR = 0.1 ; Vt = np.array([2,2,0])
while(It_index <= Max_iterrasion):
   dl_to_dTheta = np.dot(np.dot(Theta,X) - Y ,X) / len(X)
   Vt = Gamma*Vt + LR*dl_to_dTheta
   Theta=Theta - Vt
   Hypothesis= Theta[0,0] + Theta[0,1]*X[1,:] + Theta[0,2]*(X[1,:]**2)  ; Hypothesis = Hypothesis.reshape(3,1)
   Loss1 = 0.5 * np.sum((Hypothesis-Y.T)**2).mean()
   print('Loss for iteration: ',It_index,'is: ',Loss1)
   print('test:', Hypothesis)
   It_index+=1
print('New theta: ',New_theta)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#1.4.
#Repeat the process using LR=0.1, but this time with Nesterov accelerated
#gradient γ = 0 .9
print('\n\n~~~~~~~~~~~~~~ Nesterov ~~~~~~~~~~~~~~')
Theta = np.array([2,2,0]).reshape(1,3)

Max_iterrasion = 100 ; It_index = 1;
Gamma = 0.9  ;  LR = 0.1 ; Vt = Theta = np.array([2,2,0])
while(It_index <= Max_iterrasion):
   Nesterov_Theta = Theta - Gamma*Vt
   dl_to_dTheta = np.dot(np.dot(Nesterov_Theta,X) - Y ,X) / len(X)
   Vt = Gamma*Vt + LR*dl_to_dTheta
   Theta=Theta - Vt
   Hypothesis= Theta[0,0] + Theta[0,1]*X[1,:] + Theta[0,2]*(X[1,:]**2) ; Hypothesis = Hypothesis.reshape(3,1)
   Loss1 = 0.5 * np.sum((Hypothesis-Y.T)**2).mean()
   print('Loss for iteration: ',It_index, 'is: ',Loss1)
   print('test:', Hypothesis)
   It_index+=1
print('New theta: ',New_theta)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#We take the function y=5 x2+7 x−5
#2.1
#Choose random point from function.
#2.2
#Calculate the minimum with gradient descent.
#2.2.1
#Find the next point by calculating the derivative and multiplying by
#α=0.001
#2.2.2
#Draw a graph with the location of new point, using Matplotlib.pyplot
#Repeat with α=0.01 , α=0.0001
#3
#Repeat for five additional random start points.

import numpy as np
import matplotlib.pyplot as plt

for run in range(1,6):
   x = -1 + 2*np.random.rand()
   const = x
   y=5*x**2 + 7*x-5

   ax=plt.figure()
   for plot_index in range(1,3):
       if plot_index == 1:
           Alpha=0.01
       else:
           Alpha=0.0001
           x = const
       x_arr=[]
       y_arr=[]
       for  ii in range(1,10000):
           y=5*x**2 + 7*x-5
           x_arr.append(x)  # the minmum value is the last one in array
           x=x-Alpha*(10*x+7)
           y_arr.append(y)
       x_arr=np.array(x_arr)
       y_arr=np.array(y_arr)
       ax=plt.subplot(2,1,plot_index)
       ax.plot(x_arr,y_arr,'bo')
       ax.set(title = 'Alpha = {}'.format(str(Alpha)))