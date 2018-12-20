"""
EYAL YAFFE
"""


#%%
#question 1
def reverse(string):
    return string[::-1]
#test
#reverse("I am testing") #pass

#%%
#question 2
def overlapping(ls1, ls2):
    ans = []
    for element in ls1:
        if element in ls2:
            ans.append(element)
    if len(ans)==0:
        return False
    else:
        return True
#test
#print(overlapping([1,2,3],[4,5,6])) # return false and passed
#print(overlapping([1,2,3],[4,1,6])) # return true and passed

#%%
#question 3
import numpy as np
x = np.ones((4,4))
x[1:-1,1:-1]=0
#test
#print(x)        

#%%
#question 4
import pandas as pd
url="https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv"
df=pd.read_csv(url,delimiter='\t',encoding='utf-8')
pd.set_option('display.expand_frame_repr', False)
df['item_price']=df['item_price'].str.replace('$','')
df['item_price']=df.item_price.astype(float)
print(df[df.item_price>10].item_price.nunique())

#%%
#question5
X=np.array([[0,1,2],[0,1,4],[1,1,1]]).T
Y=np.array([1,3,7])
Teta=np.array([2,2,0]).reshape(3,1)
alpha=1
def gradient_descent(x,y,teta, iterations,alpha):
    for i in range(iterations):
        Y_predicted=x @ teta
        error=Y_predicted-y
        teta=teta-alpha*(x.T@error/len(x))
        print("iteration: {}, loss: {}".format(i,np.sum(error**2)))       
gradient_descent(X,Y,Teta,3,0.1)
gradient_descent(X,Y,Teta,3,1)

"""
we can see that the lost in alpha=0.1 converges and when alpha is 1 its not.
the reason for not converging is because the weights (teta) is strongly effected by the learning rate.
we can conclude that function "jumps" to a point which is far from the local minimum   

"""

#%%
#question 6
""" 
an object is an instance of a class. 
it is an abstract representation of aggregrated data unit.
it includes fields(attributes) and methods.
object are held in the heap section of the program memory and are deallocated by python garbage collector.  
"""
#%%
#question7
class animal:
    def __init__(self, number_of_legs):
        self.number_of_legs=number_of_legs
    
    def voice(self):
        print("I am an animal!")
        
class cow(animal):
    def __init__(self):
        super().__init__(4) #calling animal constructoer   
    def voice(self):        #overriding voice
        print("I am a cow!")        
      
class kengeroo(animal):
    def __init__(self):
        super().__init__(2)   
    def voice(self):        #overriding voice
        print("I am a kengeroo!")        

class snake(animal):
    def __init__(self):
        super().__init__(0)   
    def voice(self):
        print("I am a snake!")        

        
#%%
#question 8 - KNN
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data = load_iris(url)
x_train, x_test, y_train, y_test = train_test_split(data[0],data[1],test_size=0.2,random_state=42)
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train, y_train)

def findModelAccuracy(results,y_test):
    counter=0
    y_test=y_test.tolist()
    for i in range(len(results)):
        if results[i]==y_test[i]:
            counter=counter+1
    return counter/(len(results))    

test_answers=[]
for i in range(len(x_test)):
    temp_prediction=neigh.predict([x_test[i]])[0]
    test_answers.append(temp_prediction)
print(findModelAccuracy(test_answers,y_test)) #100%! :-) 

#%%
#question 8 - logisitc regression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data = load_iris(url)
x_train, x_test, y_train, y_test = train_test_split(data[0],data[1],test_size=0.2,random_state=42)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(x_train, y_train)
clf.predict(x_test[:2, :])
clf.predict_proba(x_test[:2, :]) 

#%%
#question 9
# I don't know
#%%
#question 10
"""
A[-1 0] B[0.5 0] C[0 1] D[0 -1]
 [0 -1]  [0 0.5]  [1 0]  [1  0]
"""