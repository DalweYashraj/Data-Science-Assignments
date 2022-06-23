#!/usr/bin/env python
# coding: utf-8

# # Assignment on Multiple Regression_50_Startups

# In[1]:


#Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import seaborn as sns


# In[2]:


startup = pd.read_csv('50_Startups.csv')
startup.head()


# # EDA & Visualization

# In[3]:


startup.info()


# In[4]:


startup.describe()


# In[5]:


startup.isna().sum()


# In[6]:


startup.shape


# In[7]:


startup.corr()


# In[8]:


sns.set_style(style='darkgrid')
sns.pairplot(startup)


# In[9]:


#Normalizing Data
startup1=startup


# In[10]:


# Drop non numerical column
startup1.drop(['State'],axis=1,inplace=True)
startup1.head()


# In[11]:


from sklearn.preprocessing import MinMaxScaler
norm =MinMaxScaler()
startup_norm = norm.fit_transform(startup1)


# In[12]:


startup_norm


# In[13]:


startup_df = pd.DataFrame(startup_norm)


# In[14]:


startup_df.columns=['RD_S',	'Admin_S',	'MRKT_S', 'Profit']
startup_df.head()


# # Model Building

# In[15]:


model = smf.ols('Profit ~RD_S+Admin_S+MRKT_S', data= startup_df).fit()


# In[16]:


model.params


# In[17]:


np.round(model.tvalues, 4),np.round(model.pvalues, 4)


# In[18]:


model.summary()


# In[19]:


#R squared values
(model.rsquared,model.rsquared_adj)


# In[20]:


#t and p-Values
print(model.tvalues, '\n', model.pvalues)


# # Simple Linear Regression

# In[21]:


slr_A = smf.ols('Profit~ Admin_S', data= startup_df).fit()
slr_A.summary()


# In[22]:


slr_M = smf.ols('Profit~ MRKT_S', data= startup_df).fit()
slr_M.summary()


# In[23]:


mlr_AM = smf.ols('Profit~Admin_S+ MRKT_S', data=startup_df).fit()
mlr_AM.summary()


# # Calculating VIF

# In[24]:


rsq_r = smf.ols('RD_S~ Admin_S+MRKT_S', data=startup_df).fit().rsquared
vif_r = 1/(1-rsq_r)

rsq_a = smf.ols('Admin_S ~ RD_S+MRKT_S', data=startup_df).fit().rsquared
vif_a = 1/(1-rsq_a)

rsq_m = smf.ols('MRKT_S~ RD_S+Admin_S', data=startup_df).fit().rsquared
vif_m = 1/(1-rsq_m)

d1 = {'Varriables':['RD_S','Admin_S','MRKT_S'], 'Vif':[vif_r,vif_a,vif_m]}
vif_df= pd.DataFrame(d1)
vif_df


# # All the variables have vif < 20, therfore no multicollinearty in variables. 
# ## So we will consider all the variables in model building

# # Residual Analysis

# # Test for Normality of Residuals (Q-Q Plot)

# In[25]:


import statsmodels.api as sm
sm.qqplot(model.resid,line='q') # line = 45 to draw the diagnoal line
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[26]:


list(np.where(model.resid>10))


# In[27]:


list(np.where(model.resid<-.10))


# # Residual plot for Homoscedasticity

# In[28]:


def get_standardized_values( vals ):
    return (vals - vals.mean())/vals.std() # z= (x-mu) / sigma


# In[29]:


plt.scatter(get_standardized_values(model.fittedvalues),
            get_standardized_values(model.resid))

plt.title('Residual Plot')
plt.xlabel('standarized fitted values')
plt.ylabel('standarized residual values')
plt.show()


# # Residual Vs Regressors

# #### Test for errors or Residuals Vs Regressors or independent variables or predictors
# 
# #### using Residual Regression Plots code graphics.plot_regress_exog(model,'x',fig)
# 
# #### exog = x-variable & endog = y-variable

# In[30]:


fig = plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'RD_S', fig=fig )
plt.show()


# In[31]:


fig = plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Admin_S', fig=fig )
plt.show()


# In[32]:


fig = plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'MRKT_S', fig=fig )
plt.show()


# # Model Deletion Diagnostics

# # Detecting Influencers/Outliers

# ## Cook’s Distance

# In[33]:


model_influence = model.get_influence()
(c, _)= model.get_influence().cooks_distance
c


# In[34]:


#Plot the influencers values using stem plot
fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(startup_df)), np.round(c, 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[35]:


np.argmax(c), np.max(c)


# # High Influence points

# In[36]:


fig,ax = plt.subplots(figsize=(15, 15))
influence_plot(model, ax=ax)
plt.show()


# In[37]:


k = startup_df.shape[1] # K is no. of columns

n = startup_df.shape[0] # n is no. of rows
print('no. of columns=',k,"\n",'no. of rows=', n)

# leverage cutoff value
leverage_cutoff = 3*((k + 1)/n)
print('leverage cutoff =',leverage_cutoff)


# In[38]:


startup_df[startup_df.index.isin([49])]


# In[39]:


startup_df.tail(5)


# In[40]:


startup_df.shape


# In[41]:


# Significant difference in value of 49th record, so it is a outlier, droping it
startup_new = startup_df
startup_new = startup_new.drop(startup_new.index[[49]],axis=0)


# In[42]:


startup_new.shape


# In[43]:


startup_new


# # Improving the model

# In[44]:


#Rebuild model and generate R-Squared and AIC values

model1 = smf.ols('Profit ~RD_S+Admin_S+MRKT_S', data= startup_new).fit()


# In[45]:


model1.summary()


# In[46]:


(c1, _)= model1.get_influence().cooks_distance
c1


# In[47]:


#Plot the influencers values using stem plot
fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(startup_new)), np.round(c1,3))
plt.xlabel('Row index')
plt.ylabel('Cooks distance')
plt.show()


# In[48]:


np.argmax(c1), np.max(c1)


# In[49]:


# leverage cutoff value
leverage_cutoff = 3*((4 + 1)/49)
print('leverage cutoff =',leverage_cutoff)


# In[50]:


# deleting 48th record since its cook's distance value is beyond leverage cutoff
startup_new[startup_new.index.isin([48])]


# In[51]:


startup_new = startup_new.drop(startup_new.index[[48]],axis=0)


# In[52]:


startup_new.shape


# # Rebuilding Model

# In[53]:


#Rebuild model and generate R-Squared and AIC values

model2 = smf.ols('Profit ~ RD_S+ Admin_S+ MRKT_S', data= startup_new).fit()


# In[54]:


model2.summary()


# In[55]:


(c2, _)= model2.get_influence().cooks_distance
c2


# In[56]:


#Plot the influencers values using stem plot
fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(startup_new)), np.round(c2,3))
plt.xlabel('Row index')
plt.ylabel('Cooks distance')
plt.show()


# In[57]:


np.argmax(c2), np.max(c2)


# In[58]:


# leverage cutoff value
leverage_cutoff = 3*((4 + 1)/48)
print('leverage cutoff =',leverage_cutoff)


# In[59]:


d2={'Model Name':['Model','Model1','Model2'],'Rsquared':[model.rsquared,model1.rsquared,model2.rsquared]}
table=pd.DataFrame(d2)
table


# ### Model 2 is having highest R-squared value

# # Model Prediction

# In[60]:


pred_data=pd.DataFrame({'RD_S':70000,"Admin_S":90000,"MRKT_S":140000},index=[0])
pred_data


# In[61]:


model2.predict(pred_data)

