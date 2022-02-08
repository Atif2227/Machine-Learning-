#!/usr/bin/env python
# coding: utf-8

# ### <font size=5 > <p style="color:red"> Comparative analysis of various regressors for predicting medical charges.

# ### So far
# In the previous approach we have done a good EDA, so in this approach we will only focus on MLs.
# 
# we have learnt that 'smoker','age' and 'bmi' have highest correlation with charges.

# In[92]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[93]:


df = pd.read_csv('medical.csv')
df.head()


# In[94]:


df.info()


# ### Covert object data type into categories.

# In[95]:


df2 = df[['sex', 'smoker', 'region']].astype('category')
df2


# ### Make a separate data frame for numeric data types.

# In[96]:


df3=df[['age', 'bmi', 'children', 'charges']]
df3


# ### Now join both the data frames

# In[97]:


df_new = pd.concat([df2, df3], axis=1, join='inner')
display(df_new)


# In[98]:


df_new.info()


# ### Now convert categories into numerical 

# In[99]:


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
label.fit(df_new.sex.drop_duplicates())
df_new.sex = label.transform(df_new.sex)
label.fit(df_new.smoker.drop_duplicates())
df_new.smoker = label.transform(df_new.smoker)
label.fit(df_new.region.drop_duplicates())
df_new.region = label.transform(df_new.region)
df_new.dtypes


# In[100]:


df_new


# > ### Have you noticed any problem?
# Do you know, '4' is assigned to which region. For that you will have to look up in df table. 

# ### Let's see correlations among different attributes with a heat map

# In[101]:


plt.subplots(1, 1, figsize=(8, 8))
sns.heatmap(df_new.corr(), annot=True, cmap='cool')


# # Multiple Linear Regression

# In[102]:


from sklearn.model_selection import train_test_split as holdout
from sklearn.linear_model import LinearRegression
from sklearn import metrics
x = df_new.drop(['charges'], axis = 1)
y = df_new['charges']
x_train, x_test, y_train, y_test = holdout(x, y, test_size=0.2, random_state=0)
mlr = LinearRegression()
mlr.fit(x_train, y_train)
print("Intercept:",mlr.intercept_,'\n')
print("Co-efficients:",mlr.coef_, '\n')
print("Score:",mlr.score(x_test, y_test), '\n')
y_pred = mlr.predict(x_test)
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_test,y_pred)
RMSE = np.sqrt(MSE)
print("RMSE:",RMSE)


# # Ridge Regression

# In[103]:


from sklearn.linear_model import Ridge
Ridge = Ridge(alpha=0.5)
Ridge.fit(x_train, y_train)
print("Intercept:",Ridge.intercept_, "\n")
print("Co-efficient:",Ridge.coef_, "\n")
print("Score:",Ridge.score(x_test, y_test), "\n")
from sklearn.metrics import mean_squared_error
y_pred = Ridge.predict(x_test)
MSE = mean_squared_error(y_test,y_pred)
RMSE = np.sqrt(MSE)
print("RMSE:",RMSE)


# # Lasso Regression

# In[104]:


from sklearn.linear_model import Lasso
Lasso = Lasso(alpha=0.2, fit_intercept=True, normalize=False, precompute=False, max_iter=1000,
              tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
Lasso.fit(x_train, y_train)
print("Intercept:",Lasso.intercept_, "\n")
print("Co-efficient:",Lasso.coef_, "\n")
print("Score:",Lasso.score(x_test, y_test), "\n")
from sklearn.metrics import mean_squared_error
y_pred = Lasso.predict(x_test)
MSE = mean_squared_error(y_test,y_pred)
RMSE = np.sqrt(MSE)
print("RMSE:",RMSE)


# # Polynomial Regressor

# In[130]:


from sklearn.preprocessing import PolynomialFeatures
x = df_new.drop(['charges', 'sex', 'region'], axis = 1)
y = df_new.charges
pol = PolynomialFeatures (degree = 2)
x_pol = pol.fit_transform(x)
x_train, x_test, y_train, y_test = holdout(x_pol, y, test_size=0.2, random_state=0)
Pol_reg = LinearRegression()
Pol_reg.fit(x_train, y_train)
y_train_pred = Pol_reg.predict(x_train)
y_test_pred = Pol_reg.predict(x_test)
print("Intercept:",Pol_reg.intercept_ ,'\n')
print("Coefficients:",Pol_reg.coef_, '\n')
print("Score",Pol_reg.score(x_test, y_test))


# #### Nice, we have got the best score so far.

# In[116]:


##Evaluating the performance of the algorithm
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_test_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_test_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))


# #### Root Mean Squared Error for Polynomial Regression is also lowest amont all the algorithms

# In[117]:


##Predicting the charges
y_test_pred = Pol_reg.predict(x_test)
##Comparing the actual output values with the predicted values
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})
df


# # Conclusions
# > ### 1. For this prediction 'smoker', 'age', 'bmi' are the highest importance attributes.
# > ### 2. Polynomial regression turns out to be thebest regression in this case.
