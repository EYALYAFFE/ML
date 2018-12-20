
#%%
import pandas as pd
url="http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
data=pd.read_csv(url)
test_data=data.head(68)
train_data=data.tail(500)
titles=[]
for i in range(32):
    titles.append(i)
train_data.columns=titles

#%%
import math
import numpy as np

def prepareFeaturesAndMax():
    columns=[]
    for i in range(2,32):
        columns.append(i)
    return 0,0,columns

def calcFeatureAndValue(data):
    temp_gain,max_gain, columns = prepareFeaturesAndMax()
    for i in range(len(columns)):
        mean=np.mean(data[columns[i]])
        temp_gain=calcGain(data,columns[i],mean)
        if temp_gain>max_gain:
            max_gain=temp_gain
            column=columns[i]
            value = mean
    return column, value
     
def calcGain(data, feature, value):
    impurity_father=calcImpurityOfFather(data) 
    average_impurity_of_sons=calcWeightedImpurityOfSons(data,feature,value)
    return impurity_father-average_impurity_of_sons
    
def calcImpurityOfFather(data):
    if (data.empty):
        return 0
    b_count=data[1].str.count('B').sum()
    m_count=data[1].str.count('M').sum()
    size=len(data)
    return 1-math.pow(b_count/size,2)-math.pow(m_count/size,2)

def calcWeightedImpurityOfSons(data,feature,value):
    small = pd.DataFrame()
    big = pd.DataFrame()
    for i in range(len(data)):
        row = data.iloc[[i]]
        if data.iloc[i][feature]<value:
            small = small.append(row) 
        if data.iloc[i][feature]>=value:
            big = big.append(row)
    small_impurity=calcImpurityOfFather(small)
    big_impurity=calcImpurityOfFather(big)
    tsize=len(data)
    ssize=len(small)
    bsize=len(big)
    avg_ampurity=(ssize/tsize)*small_impurity+(bsize/tsize)*big_impurity
    return avg_ampurity

class node:
    def __init__(self,data,deapth):
        self.deapth=deapth
        self.data=data
        self.isleaf=False
    def buildTree(self):
        T_F, M_or_B=allDataLabelsAreSame(self.data) 
        if (T_F):
            self.isleaf=True
            self.type=M_or_B
            print(self.type,self.deapth)
            return
        self.feature, self.value=calcFeatureAndValue(self.data)
        s1,s2=split(self.data,self.feature,self.value)
        self.left=node(s1,self.deapth+1)
        self.right=node(s2,self.deapth+1)
        self.left.buildTree()
        self.right.buildTree()
        
    def predict(self,row):
        if (self.isleaf==True):
            return self.type
        if (row.iloc[self.feature]<self.value):
            return self.left.predict(row)
        if (row.iloc[self.feature]>=self.value):
            return self.right.predict(row)
        
def allDataLabelsAreSame(data):
    b_count=data[1].str.count('B').sum()
    m_count=data[1].str.count('M').sum()
    size=len(data)
    if (size==b_count):
        return True, "B"
    if (size==m_count):
        return True, "M"
    return False,"NONE"

def split(data,feature,value):
    small = pd.DataFrame()
    big = pd.DataFrame()
    for i in range(len(data)):
        row = data.iloc[[i]]
        if data.iloc[i][feature]<value:
            small = small.append(row) 
        if data.iloc[i][feature]>=value:
            big = big.append(row)
    return small,big
    
def displayAccuracy(model,test_data):
    counter=0
    for i in range(len(model)):
        if (model[i]==test_data[i]):
            counter+=1
    return (counter/(len(model)))*100

root=node(train_data,0)
root.buildTree()
model=[]
for i in range(len(test_data)):
    model.append(root.predict(test_data.iloc[i]))
    
test_data_as_list=list(test_data['M'])
print(displayAccuracy(model,test_data_as_list))
