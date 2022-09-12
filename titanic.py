#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the packages 
import numpy as np,pandas as pd,sklearn
import warnings
warnings.filterwarnings("ignore")


# In[38]:


titanic_data = pd.read_csv("titanic.csv")
#titanic_data.head()


# In[39]:


titanic_data.columns


# In[40]:


#To check the count of  missing values
titanic_data.isnull().sum()


# In[41]:


#print("No of passengers travelling in ship or in original data:"+str(len(titanic_data)))


# In[42]:


#Data Visualization 
import matplotlib.pyplot as plt,seaborn as sns
titanic_data["Age"].plot.hist()
#plt.show()


# In[43]:


sns.boxplot(x="Embarked",y="Age",data=titanic_data)
#plt.show()


# In[44]:


sns.boxplot(x="Sex",y="Age",data=titanic_data)
#plt.show()


# In[45]:


#drop the major missing values
titanic_data.drop("Cabin",axis=1,inplace=True)


# In[46]:


titanic_data.isnull().sum()


# In[47]:


#Fill the values of Age column
titanic_data["Age"].fillna((titanic_data["Age"].mean()),inplace=True)


# In[48]:


titanic_data.isnull().sum()


# In[49]:


#To drop all null values
titanic_data.dropna(inplace=True)


# In[50]:


titanic_data.isnull().sum()


# In[51]:


titanic_data.info()


# In[52]:


#To create dummies
sex = pd.get_dummies(titanic_data["Sex"],drop_first=True)
sex.head()


# In[53]:


Pcl = pd.get_dummies(titanic_data["Pclass"],drop_first=True)
Pcl.head()


# In[54]:


embark = pd.get_dummies(titanic_data["Embarked"],drop_first=True)
embark.head()


# In[55]:


titanic_data = pd.concat([titanic_data,sex,Pcl,embark],axis=1)
titanic_data.head(2)


# In[59]:


#Drop the unnecessary columns
titanic_data.drop(['Sex','Embarked','Pclass','Name','Ticket','PassengerId'],
                  axis=1,inplace =True)


# In[57]:


titanic_data['SibSp'].value_counts()


# In[ ]:





# In[60]:


titanic_data.head(25)


# In[58]:


titanic_data['Parch'].value_counts()


# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


#Training and Testing data
X = titanic_data.drop('Survived',axis=1)
y = titanic_data["Survived"]


# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


#random state is basically used for reproducing your problem every time
a,b = np.arange(10).reshape(5,2),range(5)
b


# In[24]:


train_test_split(a,b)


# In[25]:


train_test_split(a,b)


# In[26]:


train_test_split(a,b,random_state=1)


# In[27]:


train_test_split(a,b,random_state=1)


# In[28]:


#Splitting the data into training and testing set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=1)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train) #Estimators 


# In[29]:


#predictors
predictions = logmodel.predict(X_test)


# In[30]:


#Metrics
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
#print(accuracy_score(y_test,predictions)*100)


# In[31]:


#print(confusion_matrix(y_test,predictions))


# In[32]:


#print(classification_report(y_test,predictions))


# In[33]:

def survive(arr):
    predictions = logmodel.predict(arr)
    return predictions


# In[ ]:





# In[36]:


#!pip install flask-ngrok


# In[37]:


'''from flask_ngrok import run_with_ngrok
from flask import Flask,jsonify
app = Flask(__name__)
run_with_ngrok(app) #starts ngrok when app is running
@app.route("/<float:Age>/<int:SibSp>/<int:Parch>/<float:Fare>/<Gender>/<int:Pclass>/<Place>")
def home(Age,SibSp,Parch,Fare,Gender,Pclass,Place):
  p = []
  p +=[Age,SibSp,Parch,Fare]
  if Gender.casefold() == "m":
    p+=[1]
  else:
    p+=[0]
  if Pclass == 2:
    p+=[1,0]
  elif Pclass == 3:
    p+=[0,1]
  else:
    p+=[0,0]
  if Place.casefold() == "queenstown":
    p+=[1,0]
  elif Place.casefold() == "southampton":
    p+=[0,1]
  else:
    p+=[0,0]
  arr = np.array([p])
  predict = logmodel.predict(arr)
  if predict == [1]:
    result = {'result':'Survived'}
  else:
    result = {'result':'Not Survived'}
  return jsonify(result)
app.run()'''


# In[ ]:




