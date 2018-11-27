#%%
#ex1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
#ex2+ex3+ex4
url="https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv"
chipo=pd.read_csv(url,delimiter='\t',encoding='utf-8')
pd.set_option('display.expand_frame_repr', False)
print(chipo.head(20))

#%%
#ex5
print(len(chipo.index))
#%%
#ex6
print(len(chipo.columns))

#%%
#ex7
print(chipo.columns)
#%%
#ex8
print(chipo.index)

#%%
#ex9
chipo9=chipo.groupby(['item_name']).size().argmax()
print(chipo9)
#%%
#ex10 How many items were ordered?
Total = chipo['quantity'].sum()
print(Total)

#%%
#ex11 
print(chipo.groupby('choice_description').choice_description.count().argmax())

#%%
#ex12
#same as 10

#%%
#ex13
chipo['item_price']=chipo['item_price'].str.replace('$','')
print(chipo)
chipo['item_price']=chipo.item_price.astype(float)
print(chipo.dtypes)

#%%
#ex14
print(chipo[["quantity", "item_price"]].product(axis=1).sum())

#%%
#ex15
#same as previous questions

#%%
#ex16
print(chipo['item_price'].sum()/len(chipo.index))

#%%
#ex17
print(chipo.item_name.value_counts().count())

#%%
#part 2. Filtering & Sorting
#3
url="https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv"
chipo=pd.read_csv(url,delimiter='\t',encoding='utf-8')
pd.set_option('display.expand_frame_repr', False)
print(chipo.head(10))

#%%
#4
print(chipo[chipo.item_price > 10].item_price.count())
#%%
#5
print(chipo[chipo.quantity == 1][['item_name','item_price']].groupby('item_name'))

#%%
#6
print(chipo.sort_values(by=['item_name']))
#%%
#7
print(chipo.sort_values(by=['item_price']).tail(1).quantity)

#%%
#8
print(chipo.groupby("item_name").item_name.count()['Veggie Salad Bowl'])

#%%
#9
canned_soda = chipo['item_name'] == "Canned Soda"
qisgto = chipo['quantity'] > 1
print(chipo[canned_soda & qisgto])
print(len(chipo[canned_soda & qisgto].index))

#%%
#PART 3
url="https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user"
users=pd.read_csv(url,delimiter='|',encoding='utf-8')
pd.set_option('display.expand_frame_repr', False)
print(users.head(10))

#%%
#4
print(users.groupby("occupation").age.mean())

#%%
#5
mans = users[users.gender == 'M']
ans = mans.groupby('occupation').gender.count()
print(ans)

#%%
#6
print(users.groupby('occupation').age.max())
print(users.groupby('occupation').age.min())

#%%
#7
man=users[users.gender == 'M']
print(mans)
print(mans.groupby('occupation').age.mean())
woman=users[users.gender == 'F']
print(woman)
print(woman.groupby('occupation').age.mean())

#%%
#8
man=users[users.gender == 'M']
df_of_mans=mans.groupby('occupation').count().user_id
print(df_of_mans)
woman=users[users.gender == 'F']
df_of_womans=woman.groupby('occupation').count().age
print(df_of_womans)
temp=pd.concat([df_of_mans, df_of_womans], axis=1, join='inner')
print(temp)
temp["sum"] = temp.user_id+temp.age
temp["man_percentage"] = temp['user_id']/temp['sum']
temp["man_percentage"] = temp["man_percentage"]*100
print(temp)

    
#%%
#part 4. Merge
raw_data_1 = {
'subject_id': ['1', '2', '3', '4', '5'],
'first_name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
'last_name': ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']}
raw_data_2 = {
'subject_id': ['4', '5', '6', '7', '8'],
'first_name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']}
raw_data_3 = {
'subject_id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],
'test_id': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]}    
data1=pd.DataFrame(raw_data_1)
data2=pd.DataFrame(raw_data_2)
data3=pd.DataFrame(raw_data_3)

print(data1)    
print(data2)    
print(data3)  

#%%
#question 4 Join the two dataframes along rows and assign all_data
all_data=pd.concat([data1,data2],ignore_index=True)
print(all_data)
#%%
#question 5
all_data_col = pd.concat([data1, data2], axis=1, sort=False)
print(all_data_col)
#%%
#question 7
print(all_data)
print(data3)
ans7 = pd.merge(all_data, data3)
print(ans7)
#%%
#question 8
ans8 = pd.merge(all_data,data3, on ="subject_id",how='inner')
print(ans8)

#%%
#PART 5
url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris=pd.read_csv(url,delimiter=',',encoding='utf-8')
pd.set_option('display.expand_frame_repr', False)
print(iris.head(10))
#%%
#4
iris.insert(5, "Team") 
print(iris)


























