
#%%
#1.1
import numpy as np

X=np.array([[0,1,2],[0,1,4],[1,1,1]]).T
Y=np.array([1,3,7])
Teta=np.array([2,2,0]).reshape(3,1)
alpha=1
iterations=2
def gradient_descent(x,y,teta):
    for i in range(iterations):
        Y_predicted=x @ teta
        error=Y_predicted-y
        teta=teta-alpha*(x.T@error/len(x))
        print("iteration: {}, loss: {}".format(i,np.sum(error**2)))       
gradient_descent(X,Y,Teta)

#%%
#1.3
X=np.array([[0,1,2],[0,1,4],[1,1,1]]).T
Y=np.array([1,3,7])
Teta=np.array([2,2,0]).reshape(3,1)
alpha=0.1
gamma=0.9
iterations=50
def gradient_descent_momentum(x,y,teta,vt):
    for i in range(iterations):
        Y_predicted=x @ teta
        error=Y_predicted-y
        vt=gamma*vt+alpha*(x.T@error/len(x))
        teta=teta-vt
        print("iteration: {}, loss: {}".format(i,np.sum(error**2)))       
gradient_descent_momentum(X,Y,Teta,0)

#%%
#1.4
X=np.array([[0,1,2],[0,1,4],[1,1,1]]).T
Y=np.array([1,3,7])
Teta=np.array([2,2,0]).reshape(3,1)
alpha=0.1
gamma=0.9
iterations=10
def gradient_descent_nestrov(x,y,teta,vt):
    for i in range(iterations):
        Y_predicted=x@(teta-vt*gamma)
        error=Y_predicted-y
        vt=gamma*vt+alpha*(x.T@error/len(x))
        teta=teta-vt
#        if i%100==0:
        print("iteration: {}, loss: {}".format(i,np.sum(error**2)))       
gradient_descent_nestrov(X,Y,Teta,0)

#%%
#2.2
def derive(x):
    





