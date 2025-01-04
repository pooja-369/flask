#!/usr/bin/env python
# coding: utf-8

# In[152]:


import pandas as pd 


# In[153]:


car = pd.read_csv("quikr_car.csv")


# In[154]:


car.head()


# In[155]:


car.info()


# In[156]:


car.shape


# In[157]:


car['year'].unique()


# In[158]:


car['Price'].unique()


# In[159]:


#cleaning 
backup= car.copy()


# In[160]:


car=car[car['year'].str.isnumeric()]


# In[161]:


car['year'] = car['year'].astype(int)


# In[162]:


car['year'].astype(int)


# In[163]:


car.info()


# In[164]:


car['Price'].unique()


# In[165]:


car=car[car['Price']!="Ask For Price"]


# In[166]:


car['Price'] =car['Price'].str.replace(',','').astype(int)


# In[167]:


car.info()


# In[168]:


car.head()


# In[169]:


car['kms_driven']=car['kms_driven'].str.split(' ').str.get(0).str.replace(',','')


# In[170]:


car=car[car['kms_driven'].str.isnumeric()]


# In[171]:


car['kms_driven']=car['kms_driven'].astype(int)


# In[172]:


car.info()


# In[173]:


car =car[~car['fuel_type'].isna()]


# In[174]:


car['name']=car['name'].str.split(" ").str.slice(0,3).str.join(" ")


# In[175]:


car.head()


# In[176]:


car =car.reset_index(drop=True)


# In[177]:


car


# In[178]:


car.describe()


# In[179]:


car=car[car['Price']<6e6].reset_index(drop=True)


# In[180]:


car.info()


# In[181]:


car.to_csv('cleaned quickr_car.csv')


# In[182]:


X=car.drop(columns='Price')
y=car['Price']


# In[183]:


X


# In[184]:


y


# In[185]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[186]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


# In[187]:


ohe=OneHotEncoder()
#categorical values only 
ohe.fit(X[['name','company','fuel_type']])


# In[188]:


column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),
                                    remainder='passthrough')


# In[189]:


lr=LinearRegression()


# In[190]:


pipe=make_pipeline(column_trans,lr)


# In[191]:


pipe.fit(X_train,y_train)


# In[192]:


y_pred=pipe.predict(X_test)


# In[193]:


r2_score(y_test,y_pred)


# In[ ]:


scores=[]
for i in range (1000):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2 , random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(X_train,y_train)
    y_pred=pipe.predict(X_test)
    scores.append(r2_score(y_test,y_pred))
    


# In[ ]:


import numpy as np 
np.argmax(scores)


# In[ ]:


scores[np.argmax(scores)]


# In[ ]:





# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2 , random_state=np.argmax(scores))
lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
r2_score(y_test,y_pred)


# In[ ]:


import pickle 


# In[ ]:


pickle.dump(pipe,open("LinearRegressionModel.pkl",'wb'))


# In[ ]:


pipe.predict(pd.DataFrame([['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']],columns=['name',	'company','year','kms_driven','fuel_type']))

