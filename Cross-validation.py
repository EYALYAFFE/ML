
#%%
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import KFold

url = "pima-indians-diabetes.csv"
data = pd.read_csv(url)

X, y = data.iloc[:, :-1], data.iloc[:, -1]
kwargs = dict(test_size=0.22, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, **kwargs)
X_train.columns=['a','b','c','d','e','f','g','h']
X_test.columns=['a','b','c','d','e','f','g','h']

kf = KFold(n_splits=10)
kf.get_n_splits(X_train, y_train)
for train_index, test_index in kf.split(X, y):
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]


Train=X_train.assign(Class = y_train)
Zero_class=Train.loc[Train.loc[:,'Class']==0]
One_class=Train.loc[Train.loc[:,'Class']==1]
mean_std_table=np.arange(32,dtype='f').reshape((2, 16))
mean_std_table[0][0]=Zero_class.a.mean()
mean_std_table[0][1]=Zero_class.a.std()
mean_std_table[0][2]=Zero_class.b.mean()
mean_std_table[0][3]=Zero_class.b.std()
mean_std_table[0][4]=Zero_class.c.mean()
mean_std_table[0][5]=Zero_class.c.std()
mean_std_table[0][6]=Zero_class.d.mean()
mean_std_table[0][7]=Zero_class.d.std()
mean_std_table[0][8]=Zero_class.e.mean()
mean_std_table[0][9]=Zero_class.e.std()
mean_std_table[0][10]=Zero_class.f.mean()
mean_std_table[0][11]=Zero_class.f.std()
mean_std_table[0][12]=Zero_class.g.mean()
mean_std_table[0][13]=Zero_class.g.std()
mean_std_table[0][14]=Zero_class.h.mean()
mean_std_table[0][15]=Zero_class.h.std()
mean_std_table[1][0]=One_class.a.mean()
mean_std_table[1][1]=One_class.a.std()
mean_std_table[1][2]=One_class.b.mean()
mean_std_table[1][3]=One_class.b.std()
mean_std_table[1][4]=One_class.c.mean()
mean_std_table[1][5]=One_class.c.std()
mean_std_table[1][6]=One_class.d.mean()
mean_std_table[1][7]=One_class.d.std()
mean_std_table[1][8]=One_class.e.mean()
mean_std_table[1][9]=One_class.e.std()
mean_std_table[1][10]=One_class.f.mean()
mean_std_table[1][11]=One_class.f.std()
mean_std_table[1][12]=One_class.g.mean()
mean_std_table[1][13]=One_class.g.std()
mean_std_table[1][14]=One_class.h.mean()
mean_std_table[1][15]=One_class.h.std()



#%%
import math
def calculateProbability(x, mean, stdev):
	exponent=math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1/(math.sqrt(2*math.pi)*stdev))*exponent

def calcProbabilty(X_test_row):
    zero_probabilites=[]
    ones_probabilties=[]
    for i in range(len(X_test_row)):
        temp_pzero =  calculateProbability(X_test_row.iloc[i],mean_std_table[0][2*i],mean_std_table[0][2*i+1])   
        zero_probabilites.append(temp_pzero)
    for i in range(len(X_test_row)):
        temp_pone =  calculateProbability(X_test_row.iloc[i],mean_std_table[1][2*i],mean_std_table[1][2*i+1])   
        ones_probabilties.append(temp_pone)
    zeros_product=product(zero_probabilites)
    ones_product=product(ones_probabilties)
    if zeros_product>ones_product:
        return 0
    else:
        return 1

def product(mylist):
    ans=1
    for i in range(len(mylist)):
        ans=ans*mylist[i]
    return ans
              
def findModelAccuracy(model,y_test):
    counter=0
    y_test=y_test.tolist()
    for i in range(len(model)):
        if model[i]==y_test[i]:
            counter=counter+1
    return counter/(len(model))    
from sklearn.model_selection import KFold
test_results=[]
for i in range(0,len(X_test)):
    temp_result = calcProbabilty(X_test.iloc[i,:]) #returns 1 or 0
    test_results.append(temp_result)
accuracy=findModelAccuracy(test_results, y_test)
print(accuracy)


