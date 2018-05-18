
# coding: utf-8

# In[41]:

import pandas as pd
import numpy as np
from sklearn import metrics, feature_selection
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.plotly as py
import pytest
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split



# In[2]:

df=pd.read_csv('2010 Federal STEM Education Inventory Data Set.csv',header=1,low_memory=False)
#Removing last row of total
df=df[:-1]


# In[3]:

columns = pd.DataFrame(df.columns.tolist())
columns.loc[columns[0].str.startswith('Unnamed:'), 0] = np.nan
columns[0] = columns[0].fillna(method='ffill')
df.columns=columns
df.columns=pd.MultiIndex.from_tuples(df.columns)



# # Step1

# In[4]:

df_req=df[[' C1) Funding FY2008 ',' C2) Funding FY2009 ']]
df_req=df_req.rename(index=str,columns={' C1) Funding FY2008 ':'FY2008',' C2) Funding FY2009 ':'FY2009'})
df_req=df_req.replace(' -   ',np.nan)
#df_req


# In[5]:

df_req['FY2008']=df_req['FY2008'].astype(float)
df_req['FY2009']=df_req['FY2009'].astype(float)
df_req=df_req.dropna(axis=0)
df=df.iloc[df_req.index]


# In[6]:



# In[7]:

#Part 1
df_req['% growth']=((df_req['FY2009']-df_req['FY2008'])/df_req['FY2008'])*100
#Part 2
df_req['label']=np.where(df_req['% growth']>=0,1,0)


# In[8]:




# # Step 2

# In[9]:

#Part 1
funding_col=[' C1) Funding FY2008 ',' C2) Funding FY2009 ', ' C3) Funding FY2010 ','Index Number']
non_funding_col=list(set(df.columns)-set(funding_col))
for i in non_funding_col:
    #Percentage composition of each attribute value in a column
    dff=pd.DataFrame(df[i]).apply(pd.Series.value_counts)/len(df)
    #dff.plot(kind='bar')
    #plt.show()
    #plt.close()


# In[10]:

#Part 2
dff=pd.DataFrame()
tye=type(pd.DataFrame())
for i in non_funding_col:
    if type(df[i])==type(pd.DataFrame()):
        num_cols=len(df[i].columns)
        for j in range(num_cols):
            e=pd.get_dummies(df[i].iloc[:,j])
            if e.empty:
                continue
            e=e.iloc[:,0]
            dff[str(i)+'_'+str(e.name)]=e
            score=metrics.mutual_info_score(e, df_req['label'])
            print(i," ",e.name,' ',score)
            print()
    else:
        e=pd.get_dummies(df[i]).iloc[:,0]
        dff[str(i)+'_'+str(e.name)]=e
        score=metrics.mutual_info_score(e, df_req['label'])
        print(i,' ',score)
        print()


# # Step 3

# In[11]:

#Part 1
def get_test_size(szes):
	return int(0.3*szes)

tst_size=get_test_size(szes=len(dff))
train, test ,label_train, label_test= train_test_split(dff,df_req.iloc[:,3], test_size=tst_size)

print(len(train))
# In[12]:

c=[]
for f in train.columns:
    if(any(x in f for x in set(('[', ']', '<')))):
        f=f.replace('<','')
        print(f)
    c.append(f)
train.columns=c
test.columns=c


# In[14]:

#Part 2
model = XGBClassifier()
model.fit(train, label_train)


# In[15]:

label_pred=model.predict(test)



# In[16]:

# In[17]:

def get_thresh(model,train,test,label_test,label_train):
    if (len(test)>len(train)) or (len(label_test)>len(label_train)):
        raise TypeError('Invalid train and test size')
    model1 = XGBClassifier()
    if type(model)!=type(XGBClassifier()):
        raise TypeError('Invalid model passed')
    if (pd.DataFrame(label_train).shape[1]>1) or (pd.DataFrame(label_test).shape[1]>1):
    	raise TypeError('Multiple columns in label, Invalid shape.')
    max_score=0
    thrsh=0
    thresholds = np.sort(model.feature_importances_)
    for thresh in thresholds:
        selection = feature_selection.SelectFromModel(model, threshold=thresh,prefit=True)
        select_X_train = selection.transform(train)
        selection_model = XGBClassifier()
        selection_model.fit(select_X_train, label_train)
        select_X_test = selection.transform(test)
        y_pred = selection_model.predict(select_X_test)
        scr=metrics.roc_auc_score(label_test,y_pred)
        if(scr>max_score):
            max_score=scr
            thrsh=thresh
    return thrsh

# In[18]:

thrsh=get_thresh(model,train,test,label_test,label_train)

# In[19]:

selection = feature_selection.SelectFromModel(model, threshold=thrsh,prefit=True)
select_X_train = selection.transform(train)
selection_model = XGBClassifier()
selection_model.fit(select_X_train, label_train)
select_X_test = selection.transform(test)
y_pred = selection_model.predict(select_X_test)
print(metrics.roc_auc_score(label_test,y_pred))


# In[20]:

feature_idx = selection.get_support()
feature_name = dff.columns[feature_idx]
print('Number of features selected: ',select_X_train.shape[1])


# In[21]:

#The selected features
print(feature_name)


# In[45]:

#Part 3

def test_get_test_size():
	assert get_test_size(200)==60

def test_train_test_get_thresh():
	train, test ,label_train, label_test= train_test_split(dff,df_req.iloc[:,3], test_size=0.7)
	assert get_thresh(model,train,test,label_test,label_train)

def test_model_get_thresh():
	train, test ,label_train, label_test= train_test_split(dff,df_req.iloc[:,3], test_size=0.3)
	assert get_thresh(0,train,test,label_test,label_train)

def test_label_get_thresh():
	train, test ,label_train, label_test= train_test_split(dff,df_req.iloc[:,3], test_size=0.3)
	assert get_thresh(model,train,test,test[:3],label_train)	
# In[35]:



# In[ ]:



