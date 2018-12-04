import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
#1
X=pd.DataFrame([[31, 22], [22, 21], [40, 37], [26, 25]], columns=['Older sibling', 'Younger sibling'])
Y=pd.Series([2, 3, 8, 12])
Teta=np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),Y)
#Test
#print(Teta)
#%%
#1.2
X["Ones"] = [1,1,1,1]
print(X)
Teta=np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),Y)
#Test
print(Teta)

#%%
#PART 2
#2.1
df = pd.read_csv('data_for_linear_regression.csv')
#print(df)

#%%
#2.2
data=df.values
print(data)

#%%
#2.3
plt.scatter(data[:,0], data[:,1],alpha=1)

#%%
#2.4
X=data[0:200,0]
Z=np.ones((200, 1))
X=np.hstack((X.reshape(200,1),Z))
Y=data[0:200,1]
Teta=np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),Y)
print(Teta)

#%%
#2.5
plt.scatter(X[:,0],Y,alpha=1)
x_range=np.linspace(np.min(X[:,0]),np.max(X[:,0]), 1000)
y_range=x_range*Teta[0]+Teta[1]
plt.plot(x_range,y_range,c='red')

#%%
#2.6


