
#%%
#1+2
import numpy as np
import matplotlib.pyplot as plt
import math

X1 = np.random.normal(8,2,(100,2)) ; Y1 = np.ones((100,1))
X2 = np.random.normal(13,2,(100,2)); Y2 = np.ones((100,1))
plt.scatter(X1[:,0],X1[:,1])
plt.scatter(X2[:,0],X2[:,1])
plt.show()

#%%
#3
Y=np.round(np.random.uniform(0,1,200))
X = np.vstack([X1,X2])

#%%
#4
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

#%%
#5
def prediction_of_logic(x):
    if (x>=0.5):
        return 1
    else:
        return 0
#%%
#6
def loss_function(X,Y,Hipothesis,)        
    return  
    
    
    
    
    
    

    
    
    


