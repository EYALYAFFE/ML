
###########################
#### Gradient Descent #####
###########################
import numpy as np
x=np.array([0,1,2,3],dtype=np.float32)

y=np.array([1,3,7,13],dtype=np.float32)

X=np.c_[np.ones_like(x),x,x**2]
print(X)
# regression works nicely, but this is not part of the task
#check_regression(X,y)


start = np.array([2,2,0],dtype=np.float32)

def my_model(t):
    return np.dot(X,t)

def mse_loss(res,y):
    return 0.5*((res-y)**2).mean()

def mse_loss_grad(res,y):
    return (np.dot(res-y,X))/len(X)
   

def run_gradient_descent(X,y,start,rate,epochs):
    t=start.copy()
    for epoch in range(epochs):
        print(X.shape)
        print(start.shape)
        res=np.dot(X,t)
        print(res)
        loss=0.5*((res-y)**2).mean()
        grad=(np.dot(res-y,X))/len(X)
        t=t -rate*grad
        print('epoch {}, loss {}, new t {}'.format(epoch,loss,t))


# This function is the same as the function above, with a breakdown of the matrix multiplications
def run_gradient_descent_for_loops(X,y,start,rate,epochs):
    t=start.copy()
    print(t)
    for epoch in range(epochs):
        res = np.zeros(len(X))
        print(res)
        for i in range(len(X)):
            for j in range(len(t)):
                res[i]= res[i] + X[i,j]*t[j]
        loss=0.5*((res-y)**2).mean()
        
        grads = np.zeros((len(X), len(t)))
        
        for i in range(len(X)):
            for j in range(len(t)):
                grads[i,j] = (res[i]-y[i])*X[i,j]               
        grad = grads.mean(axis=0)
        
        t=t -rate*grad
        print('epoch {}, loss {}, new t {}'.format(epoch,loss,t))
        

run_gradient_descent(X,y,start,0.01,1000)



def run_momentum_gradient_descent(X,y,start,model_function,rate,momentum_decay,epochs):
    t=start.copy()
    v=np.zeros_like(start)
    for epoch in range(epochs):
        res=np.dot(X,t)
        loss=mse_loss(res,y)
        grad=mse_loss_grad(res,y)
        v=momentum_decay*v - rate*grad
        t= t+v
        print('epoch {}, loss {}, new t {}'.format(epoch,loss,t))


run_momentum_gradient_descent(X,y,start,my_model,0.01,0.9,1000)


def run_nesterov_momentum_gradient_descent(X,y,start,model,rate,momentum_decay,epochs):
    t=start.copy()
    v=np.zeros_like(start)
    for epoch in range(epochs):
        res=np.dot(X,t)
        loss=mse_loss(res,y)
        grad=mse_loss_grad(np.dot(X,t+momentum_decay*v),y)
        v=momentum_decay*v - rate*grad
        t= t+v
        print('epoch {}, loss {}, new t {}'.format(epoch,loss,t))
        
        
run_nesterov_momentum_gradient_descent(X,y,start,my_model,0.01,0.9,1000)



