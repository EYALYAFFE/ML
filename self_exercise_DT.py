
#%%
import pandas as pd
url="forest.csv"
data=pd.read_csv('Decision Tree DataSet.csv')
feature=data['outlook']
feature=feature.to_frame()
tree=createTree(data)
print(tree)
#%%
def createTree(data):
    if    


#%%
import numpy as np

def try_div(x,y):
    if y==0: 
        return 0
    return x/y

list_of_params=[]

def iParams(data, feature):
    members=feature.iloc[:,0].unique()
    entropi=0
    for member in members:
        member_instances=feature.loc[feature.outlook == member, feature.columns.values[0]].count()
        yes_instances=data[(data[feature.columns.values[0]]== member) & (data.play == 'yes')].count()[0]
        no_instances=data[(data[feature.columns.values[0]]== member) & (data.play == 'no')].count()[0]
        comp1=-try_div(yes_instances,member_instances)*np.log2(try_div(yes_instances,member_instances))
        comp2=-try_div(no_instances,member_instances)*np.log2(try_div(no_instances,member_instances))
        entropi=comp1+comp2
        list_of_params.append(entropi)
    for n, i in enumerate(list_of_params):
        if np.isnan(i):
            list_of_params[n]=0
iParams(data, feature)
print(list_of_params)        



#%%
import numpy as np
#ans1 = (-4/7)*(np.log2(4/7))-(3/7)*np.log2(3/7)
#ans2 = (-6/7)*(np.log2(6/7))-(1/7)*np.log2(1/7)
#ans3 = (-9/14)*(np.log2(9/14))-(5/14)*np.log2(5/14)
#print(ans3)
gain_forcest = 0.94-(5/14)*0.971*2
print(gain_forcest)

#%%
class Tree:
    def __init__(self):
        self.root=node("root")
    def addNode(self,name):   
        self.root.addChild(name)        

class Node:
    def __init__(self, name):
        self.name_of_node=name
        self.list_of_childs=[]
    def addChild(self, name):
        self.list_of_childs.append(name)
        
        
        
        
        



