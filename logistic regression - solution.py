import numpy as np
import scipy.stats as stats 
from matplotlib import pyplot as plt

#np.random.seed(5)

def generate_random_clusters(n_points_in_cluster,n_clusters, std):
    center = np.random.rand(2)*4
    angles = 2*np.pi* np.linspace(0,1,n_clusters+1)[:-1]
    centers = np.c_[np.cos(angles),np.sin(angles)] + center
    noise = np.random.normal(0,std,(n_clusters,n_points_in_cluster,2))
    points = np.repeat(np.expand_dims(centers,1),n_points_in_cluster,axis=1)
    return points + noise

n_points_in_cluster = 99
n_clusters = 2
std = 0.8
data = generate_random_clusters(n_points_in_cluster,n_clusters, std)
data_points=data.reshape(-1,2)

import matplotlib.cm as cm
colors = cm.rainbow(np.linspace(0, 1, n_clusters))

nn=n_points_in_cluster

for i in range(n_clusters):
    plt.scatter(data_points[nn*i:nn*(i+1),0],data_points[nn*i:nn*(i+1),1],color=colors[i])

y=np.repeat(np.array(range(n_clusters)),n_points_in_cluster)
X = np.c_[data_points,np.ones(data_points.shape[0])]

def split_train_test(X,y,percentage_test):
    per_index=int(len(y)*(1-percentage_test))
    return X[:per_index,...],X[per_index:,...],y[:per_index],y[per_index:]

percentage_test = 0.2
indices= np.array(range(n_points_in_cluster*n_clusters))
np.random.shuffle(indices)

X_train, X_test, y_train, y_test = split_train_test(X[indices,:],y[indices],percentage_test)

def logit(x):
    return 1/(1+ np.exp(-x))

#def softmax(x):
#    exp_norm = np.exp(x-x.max())
#    return exp_norm/exp_norm.sum(axis=1)
              
EPS = 1e-7

def minus_log_likelihood(res,y):
    return -(y*np.log(res+EPS)+(1-y)*np.log(1-res+EPS)).mean()

def predict_logit(x, teta):
    return logit(np.dot(x,teta))>0.5

#def predict_softmax(x, teta):
#    return softmax(np.dot(x,teta)).argmax()


def run_gradient_descent_logit(X,y,start,rate,epochs):
    t=start.copy()  
    for epoch in range(epochs):
        res=logit(np.dot(X,t))
        loss=minus_log_likelihood(res,y)
        grad=np.dot((res-y),X)/len(X)
        t=t-rate*grad
        print('epoch {}, loss {}, new t {}'.format(epoch,loss,t))
    return t
        
start =  np.append(np.random.normal(0,0.1,(2)),0)
teta = run_gradient_descent_logit(X_train,y_train,start,0.1,100)
train_precision=(predict_logit(X_train, teta)==y_train).sum()/len(y_train)
test_precision=(predict_logit(X_test, teta)==y_test).sum()/len(y_test)
print('Train precision: {} Test precision: {}'.format(train_precision, test_precision))
print(teta)


plt.figure()
    
def create_circles(n_points_in_cluster,n_clusters, std):
    center = np.random.rand(2)*4
    print('circle center: ', center)
    angles = 2*np.pi*np.random.rand(n_clusters,n_points_in_cluster)
    
    #radii =np.array([stats.truncnorm.rvs(-0.1+i,0.1+i,size=(n_points_in_cluster)) for i in range(2,2+n_clusters)])
    arrays = []
    for i in range(n_clusters):
        arrays.append(np.random.normal(0,0.3,n_points_in_cluster)+2+i)
        
    radii = np.array(arrays)
    x_y_coords = np.array([radii*np.cos(angles)+center[0],radii*np.sin(angles)+center[1]])

    # The two swap axes will re-shuffle the dimensions to the order of n_clusters X n_points_in_cluster X Coordinates
    return np.swapaxes(np.swapaxes(x_y_coords,0,2),0,1)


circles=create_circles(n_points_in_cluster,n_clusters, std)

for i in range(circles.shape[0]):
    # plot each set of points
    plt.scatter(circles[i,:,0],circles[i,:,1],color=colors[i])
    # define coundaries of plot
    plt.xlim(circles[:,:,0].min(),circles[:,:,0].max())
    plt.ylim(circles[:,:,1].min(),circles[:,:,1].max())


data_points=circles.reshape(-1,2)
X=data_points
# create labels
y=np.repeat(np.array(range(n_clusters)),n_points_in_cluster)
#X = np.c_[data_points,np.ones(data_points.shape[0])]
indices= np.array(range(n_points_in_cluster*n_clusters))
np.random.shuffle(indices)
X_train, X_test, y_train, y_test = split_train_test(X[indices,:],y[indices],percentage_test)


def model_prediction(X,t):
    circle_feature = (X[:,0]-t[0])**2 + (X[:,1]-t[1])**2 - t[2]**2
    return logit(circle_feature)>0.5


def run_gradient_descent_circles_logit(X,y,start,rate,epochs):
    t=start.copy()  
    for epoch in range(epochs):
        res=model_prediction(X,t)
        loss=minus_log_likelihood(res,y)
        # Derivative of the circle formula for each t
        model_grad_vec = -np.c_[2*(X[:,0]-t[0]), 2*(X[:,1]-t[1]),2*t[2]*np.ones(len(X))]
        # pred - y is the combined derivative of cross entropy loss and logit (or softmax)
        # We multiply it by the chain rule, the dot sums the results among different values of X, len applies average
        grad=np.dot((res-y),model_grad_vec)/len(X)
        t=t-rate*grad
        print('epoch {}, loss {}, new t {}'.format(epoch,loss,t))
    return t


#start =  np.append(np.random.normal(0,0.1,(2)),0)
start = np.random.rand(3)*4
teta = run_gradient_descent_circles_logit(X_train,y_train,start,0.1,50)
train_precision=(model_prediction(X_train, teta)==y_train).sum()/len(y_train)
test_precision=(model_prediction(X_test, teta)==y_test).sum()/len(y_test)
print('Train precision: {} Test precision: {}'.format(train_precision, test_precision))
print(teta)
