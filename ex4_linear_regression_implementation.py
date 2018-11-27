import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

#
#%%
#1.1
X=np.array([[31,22],[22,21],[40,37],[26,25]])
y=np.array([2,3,8,12])
reg=linear_model.LinearRegression().fit(X,y)
print(reg.coef_)
print(reg.intercept_)

#%%
#1.2
X=np.array([[31,22,1],[22,21,1],[40,37,1],[26,25,1]])
y=np.array([2,3,8,12])
#Theta = (X_transpose * X)^(-1) * X_tr*y
teta = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(),X)),X.transpose()),y)
print(teta)

#%%
#2
data = pd.read_csv("data_for_linear_regression.csv")
data = data.head(200) 
plt.scatter(data.x, data.y)
plt.show()

df = data.iloc[0:10, 0:3]
X=np.array(df.x)
Y=np.array(df.y)
ones=np.ones(len(X))
x1=X
X=np.column_stack((X,ones))
plt.scatter(df.x, df.y)
plt.show()
reg2=linear_model.LinearRegression().fit(X,Y)
plt.scatter(x1,Y,alpha=0.5)
m=reg2.coef_
b=reg2.intercept_
print(m[0])
print(b)
y2=m[0]*x1+b
print(y2)

plt.plot(x1,y2,'r',label='best fit line')
plt.show()
#%%
#1.2

x=np.matrix([[1, 2], [3, 4]])
y=np.matrix([[3, 4], [2, 1]])
z=x+y
print(z)


