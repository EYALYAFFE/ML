
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#1#############################
#
import numpy as np

X=np.array([[31,22],[22,21],[40,37],[26,25]])
y=np.array([2,3,8,12])


#def linear_regression(X,y):
#    return np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(),X)),X.transpose()),y)
def linear_regression(X,y):
    dot = np.dot(X.transpose(),X)
    #print(dot)
    inv = np.linalg.inv(dot)
    #print(inv)
    inv_mul = np.dot(inv, X.transpose())
    return np.dot(inv_mul,y)

def check_regression(X,y):
    res=linear_regression(X,y)
    print('Regression result: ',res)
    print('predicted y:', np.dot(X,res))
    print('loss :', ((np.dot(X,res)-y)**2).mean())

# 3D plane crossing (0,0,0)
check_regression(X,y)

# 3rd feature: constant
X1=np.ones((4,3))
X1[:,:-1]=X
check_regression(X1,y)

# 3rd feature: x1-x2
X2=X1.copy()
X2[:,2]=X[:,0]-X[:,1]
check_regression(X2,y)

# 3rd feature: (x1-x2)^2
X3=X1.copy()
X3[:,2]=(X[:,0]-X[:,1])**2
check_regression(X3,y)

# 3rd feature: (x1-x2)^2, 4th feature constant
X4=np.ones((4,4))
X4[:,:-1]=X3
check_regression(X4,y)

#2####################################################


max_num_points = 200

#2.1:
training_set = pd.read_csv('data_for_linear_regression.csv')
2.2
# Convert all data to matrix for easy consumption
x_training_set_ = training_set['x']
x_training_set_ = x_training_set_.values
#b = training_set['b'][1:10]
#b = b.values
x_training_set = x_training_set_[0:max_num_points]
y_training_set_ = training_set['y']
y_training_set_ = y_training_set_.values
y_training_set = y_training_set_[0:max_num_points]

#2.3:

plt.title('Relationship between X and Y')
plt.scatter(x_training_set, y_training_set,  color='black')
plt.show()

#2.4:
b_ = np.ones((1,x_training_set_.shape[0]))
b = b_[:,0:max_num_points]
x_b = np.hstack((x_training_set.reshape(max_num_points, 1),b.reshape(max_num_points ,1)))
res = linear_regression(x_b, y_training_set)
print(res)

#2.5:

plt.hold
y_out = res[0]*x_training_set + res[1]*b
plt.plot(x_training_set, y_out.reshape(max_num_points,),  color='red')
plt.show()

#2.6
plt.figure()
plt.title("test group")
y_out = res[0]*x_training_set_[max_num_points:] + res[1]*b_[:,max_num_points:]
plt.plot(x_training_set_[max_num_points:], y_out.reshape(500,),  color='red')
plt.xlim((0,100))
plt.ylim(0,100)
plt.show()
plt.hold
plt.scatter(x_training_set_[max_num_points:], y_training_set_[max_num_points:],  color='black')



