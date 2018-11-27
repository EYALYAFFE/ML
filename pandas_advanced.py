import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#%%
#part3
pd.set_option('display.max_columns', None)
url="https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/US_Crime_Rates/US_Crime_Rates_1960_2014.csv"
crime=pd.read_csv(url,delimiter='\,',encoding='utf-8')
pd.set_option('display.expand_frame_repr', False)
print(crime.head(20))
#%%
#4
print(crime.dtypes)
#%%
#5
crime.Year = pd.to_datetime(crime.Year, format='%Y')
print(crime.dtypes)

#%%
#6
crime=crime.set_index('Year')
#%%
#7
crime=crime.drop(columns=['Total'],axis=1)

#%%
#8
print(crime.groupby((crime.index.year//10)*10).sum())
#%%
#9
#TBD

#%%
#PART 2
baby_names = pd.read_csv("NationalNames.csv")
print(baby_names.head(2000000)) 


#%%
#ex6
print(baby_names.groupby('Gender').count())

#%%
#ex7

#%%
#ex8

#%%
#ex9
print(baby_names['Name'].value_counts().argmax())

#%%
#ex10
#How many different names have the least occurrences?
names=baby_names.groupby('Name').count()
names_are_one=names.loc[names['Count'] == 1]
print(names_are_one.count())

#%%
#ex11
print(names.median())

#%%
#ex12
print(names.std())

#%%
#ex13
print(names.describe())

#%%
#PART 3
pd.set_option('display.max_columns', None)
url="https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv"
chipo=pd.read_csv(url,delimiter='\t',encoding='utf-8')
pd.set_option('display.expand_frame_repr', False)
print(chipo.head(10))

#%%
#5
top_bought=chipo.groupby('item_name').size()
top_5_items=top_bought.sort_values('index',ascending=False).head(5)
top_5_items=top_5_items.to_frame()
top_5_items.T.plot(kind = "hist",bins=30)

#%%
#PART 4



