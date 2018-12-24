
#%%
import pandas as pd
import numpy as np
url="http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
data=pd.read_csv(url)
data=data.values
train_data=data[0:100,:] #this is the train data
test_data=data[100:568,:] #this is the test data
#%%
import math
import random

def chooseRandomNumberOfFeatures(num_of_features):
    A = list(range(2,num_of_features))
    k = math.floor(math.sqrt(num_of_features))
    m = 1
    samples = []
    while len(samples) < m:
        samples=random.sample(A, k)
    return samples    

def subsample_create(old_data):
    indexs_of_data=[]
    new_data=np.empty([0,32],dtype='float')
    size=len(old_data)
    for i in range(size):
        indexs_of_data.append(random.randint(0,size))
    for i in range(size):
        new_data = np.vstack([new_data,data[indexs_of_data[i]]])
    return new_data

def calcGini(data,feature,mean):
    small=np.empty([0, 32], dtype=float)    #just an empty array
    big=np.empty([0, 32], dtype=float)      #another an empty array
    size_of_data=(len(data))                
    for row_index in range(size_of_data):
        row = data[row_index,:]
        if data[row_index,feature]<mean:
            small = np.vstack([small,row])
        if data[row_index,feature]>=mean:
            big = np.vstack([big,row])
    small_impurity=calcImpurity(small)
    big_impurity=calcImpurity(big)
    tsize=len(data)
    ssize=len(small)
    bsize=len(big)
    avg_ampurity=(ssize/tsize)*small_impurity+(bsize/tsize)*big_impurity
    return avg_ampurity

def calcImpurity(data):
    if (data.size==0):
        return 0
    b_count=len(data[data[:,1]=='B'])
    m_count=len(data[data[:,1]=='M'])
    size=len(data)
    return 1-math.pow(b_count/size,2)-math.pow(m_count/size,2)

def calcFeatureAndValue(data):
    min_gini=1
    features=chooseRandomNumberOfFeatures(data.shape[1])
    for feature in features:
        for row in range(len(data)):
            temp_gini=calcGini(data,feature,data[row][feature])
            if (temp_gini<min_gini):
                min_gini=temp_gini
                column=feature
                value=data[row][feature]
    return column, value

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
        if (row[self.feature]<self.value):
            return self.left.predict(row)
        if (row[self.feature]>=self.value):
            return self.right.predict(row)
        
def allDataLabelsAreSame(data):   
    b_count=0
    m_count=0
    if (not data.size == 0):
        b_count=len(data[data[:,1]=='B'])
        m_count=len(data[data[:,1]=='M'])
    size=len(data)
    if (size==b_count):
        return True, "B"
    if (size==m_count):
        return True, "M"
    return False,"NONE"

def split(data,feature,mean):
    small=np.empty([0, 32], dtype=float)
    big=np.empty([0, 32], dtype=float)
    for i in range(len(data)):
        row=data[i]
        if data[i][feature]<mean:
            small = np.vstack([small,row])
        if data[i][feature]>=mean:
            big = np.vstack([big,row])
    return small,big
root = node(train_data,0)
root.buildTree()

def displayAccuracy(model,test_data):
    counter=0
    for i in range(len(model)):
        if (model[i]==test_data[i]):
            counter+=1
    return (counter/(len(model)))*100

def create10Trees():
    roots=[]
    for i in range(10):
        root = node(subsample_create(train_data),0)
        root.buildTree()
        roots.append(root)
        print('finished build the {} tree'.format(i))
    return roots

def predicValueFor10Trees(roots,row):
    results=[]
    for i in range(10):
        temp_result=roots[i].predict(row)
        results.append(temp_result)
    m_count=results.count('M')
    b_count=results.count('B')
    if m_count>b_count:
        return 'M'
    else:
        return 'B'

roots=create10Trees()

model=[]
for i in range(len(test_data)):
    model.append(predicValueFor10Trees(roots,test_data[i]))
    
test_data_as_list=list(test_data[:,1])
print(displayAccuracy(model,test_data_as_list))