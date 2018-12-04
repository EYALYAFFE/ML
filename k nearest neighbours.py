
#%%
#1
import numpy as np
from sklearn.datasets import load_iris
url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data = load_iris(url)

#%%
#2
def distance_function(x1,x2):
    sum = 0
    for i in range(len(x1)):
        sum += pow((x1[i]-x2[i]),2)
    return np.sqrt(sum)
#test
print(distance_function(np.array([1,1,1]),np.array([0,0,0])))
#print(distance_function(data[0][0],data[0][1]))        
#%%
#3
def getNeighborsAsindexs(k,x,dataset):
    distances=[]
    for i in range(len(dataset)):
        temp_distance = distance_function(x,dataset[i])
        distances.append(temp_distance)
    return np.argsort(distances)[:k]
#print(getNeighborsAsindexs(10,np.array([0,0,0,0]),data[0]))        
#%%
#4

def biggest(a, b, c):
    Max=0
    if b>Max:
        Max=1    
    if c>Max:
        Max=2
        if b > c:
            Max=1
    return Max

def predict(x,neighbors_y):
    counter_zero=0
    counter_one=0
    counter_two=0
    for i in range(len(neighbors_y)):
        if neighbors_y[i]==0:
            counter_zero+=1 
        if neighbors_y[i]==1:
            counter_one+=1
        if neighbors_y[i]==2:
            counter_two+=1
    return biggest(counter_zero,counter_one,counter_two)
            
#%%
#5+6
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data[0],data[1],test_size=0.2,random_state=42)

def calcYNeighborsBasedOnXindexs(indexes,dataset):
    answers=[]
    for i in range(len(indexes)):
        answers.append(dataset[indexes[i]])
    return answers    

def calcAccuracy(x_train,x_test,y_train,y_test,k):
    test_answers=[]
    for i in range(len(x_test)):
        neighbors_as_indexes=getNeighborsAsindexs(k,x_test[i],x_train)
        neighbors_as_y_values=calcYNeighborsBasedOnXindexs(neighbors_as_indexes,y_train) 
        test_answers.append(predict(x_test[i],neighbors_as_y_values))
    return test_answers       

#test
ans=calcAccuracy(x_train,x_test,y_train,y_test,3)
print(*ans, sep=' ')
print(y_test)

#%%
#PART 2









   
