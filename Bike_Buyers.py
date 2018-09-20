
# coding: utf-8

# # WORKING ON BIKE BUYERS DATASET

# In[1]:




#Importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# In[2]:


os.getcwd()


# In[3]:


#setting the path to the DataSets folder 


os.chdir("/home/akash_cheerla/Desktop/Data_Sets")


# In[4]:


os.getcwd()


# In[7]:


#Retrieving the data from .csv file

bb1 = pd.read_csv('bikebuyer_new.csv')


# In[8]:


bb1


# In[1079]:


#Checking the column names

bb1.columns


# In[1080]:


#Explaratory Data Analysis

bb1['Age'].unique()
#This unique function returns the different ages without repetition in the provided data


# In[1081]:


bb1['Age'].value_counts()
#This value_counts gives the total number of persons in the data with corresponding ages.


# In[1082]:


#to print total number of male and female individuals in the given data 
print( "------Gender wise buyers : ----------------\n" ,bb1['Gender'].value_counts() )

#to print total number of married and unmarried individuals in given data
print("\n-----Status wise buyers  : ----------------\n" ,bb1['Marital Status'].value_counts())


# In[1083]:


#Now to get the particular data of given Queiries............


bb1[bb1['Gender'].str.contains('Male') & bb1['Marital Status'].str.contains('Married')]


# In[1084]:


bb1.dtypes


# In[1085]:


#Our Aim is to predict whether the person buys the bike or not

bb1 = bb1.drop(['ID'],axis = 1)   #SInce ID doesnt affect the Bike Buying Nature


# In[1086]:


bb1


# In[1087]:


bb1.describe()   #gives statistical information of the Numeric data in the choosen dataset


# # ===> Visualization of Numeric Data

# In[1088]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
bb1['Yearly Income'].hist(bins=10)    #Visualization of number of bikebuyers per each yearly_income

plt.show()


# In[1089]:


bb1['Age'].hist(bins=10)
plt.show()


# In[1090]:


bb1['Cars'].hist(bins=10)

plt.show()


# In[1091]:


bb1.hist()


# In[1092]:


## Box Plot helps us to find the Outliers present if any

bb1.boxplot(column ='Age')  #Uni-variate Analysis



# In[1093]:


#Here we can observe many outliers so in order to get rid of those we need to normalize the data later


# In[1094]:


bb1.boxplot(column = 'Age', by = 'Marital Status')  #Bi-Variate Analysis 


# In[1095]:


#Here also we can observe many outliers


# In[1096]:


bb1.boxplot(column = 'Yearly Income', by = 'Marital Status')


# In[1097]:


pd.crosstab(bb1['Yearly Income'],bb1['Gender'],margins=True)


# In[1098]:


#Count of missing values for data set
bb1.apply(lambda x:sum(x.isnull()),axis=0)


# # ===> filling Missing Values
# 

# In[1099]:


bb1['Marital Status'].fillna('Married',inplace=True)


# In[1100]:


bb1


# In[1101]:


bb1['Gender'].fillna('Male',inplace=True)


# In[1102]:


bb1['Children'].fillna(bb1['Children'].median(),inplace=True)


# In[1103]:


bb1['Commute Distance'].fillna(bb1['Commute Distance'].mean(),inplace=True)


# In[1104]:


bb1


# In[1105]:


bb1.apply(lambda x:sum(x.isnull()),axis=0)


# # Imputing Outliers

# In[1106]:


#Outlier imputation acan be done4 through
  


# In[1107]:


bb1.boxplot(column="Yearly Income")


# In[1108]:


q1 = bb1['Yearly Income'].quantile(0.25)
q3 = bb1['Yearly Income'].quantile(0.75)
iqr = q3 - q1
fence_low = q1 - 1.5*iqr
fence_high = q3 + 1.5*iqr
bb1 = bb1.loc[(bb1['Yearly Income']>fence_low)&(bb1['Yearly Income']<fence_high)]
bb1.boxplot(column = "Yearly Income")
plt.show()


# In[1109]:


bb1.boxplot(return_type='dict')
plt.show()


# In[1110]:


bb1['Age'].unique()


# In[1111]:


bb1['Cars'].unique()


# In[1112]:


bb1['Age'].min()


# In[1113]:


bb1['Age'].max()


# In[1114]:


bb1['Yearly Income'].hist(bins=10)


# In[1115]:


bb1['Age'].hist(bins=10)


# In[1116]:


#SInce Age is well distributed there are no outliers appearing


# # Linear Regression

# # ---- hypothesis-1:   
#                          Let us consider Commute Distance doesnt affect Bike Buying Nature

# In[1117]:


#Preparing the datasets for linear regression
bike = bb1
distance = bike['Commute Distance']
distance


# In[1118]:


buy = bike['Bike Buyer']
buy


# In[1119]:


#encode the independant variables  {Categorial variable}
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x=LabelEncoder()

buy =labelencoder_x.fit_transform(buy) #index of Marital Status is 0


# In[1120]:


bike['Bike Buyer']=buy


# In[1121]:


bike


# In[1122]:


bike.boxplot(column="Bike Buyer",by="Commute Distance")


# In[1125]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
ax=plt.plot(distance,buy,'o')
plt.ylabel("bike buying")
plt.xlabel("commute distance")


# In[ ]:


mean_buy= bike["Bike Buyer"].mean()
mean_buy


# In[ ]:


plt.plot(distance,buy,'o')
plt.ylabel=("bike buying")
plt.xlabel=("Commute Distance")
plt.axhline(mean_buy,color='r',linestyle='-')
plt.show()


# In[ ]:


import statsmodels.api as sm
model=sm.OLS(distance,buy).fit()
model.summary()


# In[ ]:


import seaborn as sns
plt.plot(distance,buy,'o')
plt.ylabel=("bike buying")
plt.xlabel=("distance")
plt.axhline(mean_buy,color='r',linestyle='-')
sns.regplot(x='Commute Distance',y='Bike Buyer',data=bike,color='g')


# # Catagorical Conversion

# In[ ]:


bb1.describe()


# In[ ]:


# Catagorical Conversion

bb1.cov()



# In[ ]:


bb1.corr()


# In[ ]:


x = bb1.iloc[:,:-1]   #seperating predictors in dataset bb1 to x
y = bb1.iloc[:,11]    #seperated Target only to another dataframe y


# In[ ]:


y


# In[ ]:


x


# In[ ]:


#encode the independant variables  {Categorial variable}
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x=LabelEncoder()

x.iloc[:,0]=labelencoder_x.fit_transform(x.iloc[:,0]) #index of Marital Status is 0
x.iloc[:,1]=labelencoder_x.fit_transform(x.iloc[:,1]) #index of Gender is 1
x.iloc[:,4]=labelencoder_x.fit_transform(x.iloc[:,4]) #index of Education is 4
x.iloc[:,5]=labelencoder_x.fit_transform(x.iloc[:,5]) #index of Occupation
x.iloc[:,6]=labelencoder_x.fit_transform(x.iloc[:,6])  # index of house ownership
x.iloc[:,9]=labelencoder_x.fit_transform(x.iloc[:,9])  # index of Region

ohe=OneHotEncoder(categorical_features=[4])  # hot encoding Education column as it has more than 3 variables
x=ohe.fit_transform(x).toarray()    # converting our x to array
ohe=OneHotEncoder(categorical_features=[5])   # hot encoding Occupation
x=ohe.fit_transform(x).toarray()
ohe=OneHotEncoder(categorical_features=[9])   #hot encoding region
x=ohe.fit_transform(x).toarray()


# In[ ]:


x


# # ===>Apply Normalization

# In[ ]:


# Since we found outliers 

from sklearn import preprocessing
minmax=preprocessing.MinMaxScaler(feature_range=(0,1))
minmax.fit(x).transform(x)


# In[ ]:


x.shape


# In[ ]:


#splitting the dataset into training set and test set
from sklearn import cross_validation,neighbors
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)  #divided the available data to both training and test sets 
# so that we can feed data for it's learning and can compare the predicted output


# In[ ]:


x_train


# In[ ]:


y_train


# In[ ]:


bb1.describe()


# # ====>  kNN Clssification

# In[ ]:


#fitting classifier to training set
clf=neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)


# In[ ]:


#predicting test set results
y_pred=clf.predict(x_test)


# In[ ]:


y_pred


# In[ ]:


from sklearn.metrics import confusion_matrix
clf_cm_test = confusion_matrix(y_test,y_pred)
clf_cm_test


# In[ ]:


accuracy = clf.score(x_test,y_test)
print(accuracy * 100)


# #  ===>Decisison Tree Classifier

# In[ ]:


# decision tree classifier
from sklearn.tree import DecisionTreeClassifier

dtc_clf = DecisionTreeClassifier()
dtc_clf.fit(x_train,y_train)


# In[ ]:


#predicting
dtc_y_test=dtc_clf.predict(x_test)
dtc_y_test


# In[ ]:


dtc_y_train = dtc_clf.predict(x_train)
dtc_y_train


# In[ ]:


from sklearn.metrics import confusion_matrix
dtc_cm_test = confusion_matrix(y_test,dtc_y_test)
dtc_cm_test


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_dtc=accuracy_score(y_test,dtc_y_test)


# In[ ]:


accuracy_dtc


# #  ====>SVM

# In[ ]:


#support vector classification
from sklearn.svm import SVC
sc=SVC(kernel='rbf')
sc_classifier = sc.fit(x_train,y_train) #model bui;lding


# In[ ]:


sc_classifier


# In[ ]:


#predicting on test and train data
svc_y_test=sc_classifier.predict(x_test)
svc_y_test


# In[ ]:


svc_y_train=sc_classifier.predict(x_train)
svc_y_train


# In[ ]:


#obtain accuracy
from sklearn.metrics import accuracy_score
accuracy3= accuracy_score(y_test,svc_y_test)
accuracy4= accuracy_score(y_train,svc_y_train)


# In[ ]:



accuracy3


# In[ ]:


accuracy4


# In[ ]:


#build confusion matrix
from sklearn.metrics import confusion_matrix
svc_cm_test=confusion_matrix(y_test,svc_y_test)
svc_cm_test


# In[ ]:


svc_cm_train=confusion_matrix(y_train,svc_y_train)
svc_cm_train


# #  Random Forest

# In[ ]:


#build random forest classifier
from sklearn.ensemble import RandomForestClassifier
rmf= RandomForestClassifier(max_depth=3,random_state=0)
rf_classi = rmf.fit(x,y)


# In[ ]:


#predicting
rmf_y_test=rmf.predict(x_test)
rmf_y_test


# In[ ]:


rmf_y_train= rmf.predict(x_train)
rmf_y_train


# In[ ]:


#Bulid confusion matrix
from sklearn.metrics import confusion_matrix
rf_cm_test = confusion_matrix(y_test,rmf_y_test)
rf_cm_test


# In[ ]:


rf_cm_train=confusion_matrix(y_train,rmf_y_train)
rf_cm_train


# In[ ]:


#Accuracy_score
from sklearn.metrics import accuracy_score
Acc_rmf=accuracy_score(y_test,rmf_y_test)
Acc_rmf


# In[ ]:


Acc_rmf_train= accuracy_score(y_train,rmf_y_train)
Acc_rmf_train


# In[ ]:


#precision score on test and train
from sklearn.metrics import precision_score
rf_precision_test = precision_score(y_test,rmf_y_test,average='weighted')
print(rf_precision_test)
rf_precision_train= precision_score(y_train,rmf_y_train,average='weighted')
print(rf_precision_train)


# In[ ]:


#Calculate recall score on test and train data
from sklearn.metrics import recall_score
rf_recall_test = recall_score(y_test,rmf_y_test,average='weighted')
print(rf_recall_test)
rf_recall_train = recall_score(y_train,rmf_y_train,average='weighted')
print(rf_recall_train)


# #  Adaptive Boosting

# In[ ]:


#perform AdaBoost
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
seed=7
num_trees=30
kfold = model_selection.KFold(n_splits=10,random_state=seed)
model=AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
result = model_selection.cross_val_score(model,x_train,y_train,cv=kfold)
print(result)


# In[ ]:


#here we got array of answers above.Thsoe are accuracies of different models and we need to calculate mean to get the
#overall accuracy.....for further explanation refer notes about AdaBoost
print(result.mean())

