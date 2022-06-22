#!/usr/bin/env python
# coding: utf-8

# In[1]:


#alary_hike -> Build a prediction model for Salary_hike
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

salary=pd.read_csv("Salary_Data.csv")


# In[2]:


salary


# # EDA & Data Visualization

# In[3]:


salary.shape


# In[4]:


salary.head()


# In[5]:


salary.info()


# In[6]:


import seaborn as sns
sns.distplot(salary['YearsExperience'])


# In[7]:


import seaborn as sns
sns.distplot(salary['Salary'])


# # Correlation Analysis

# In[8]:


salary.corr()


# In[9]:


sns.regplot(x="YearsExperience",y="Salary",data=salary)


# # Model Building

# In[10]:


newdata=pd.Series([11,11.5,12,12.5,13,13.5])


# In[11]:


data_pred=pd.DataFrame(newdata,columns=['YearsExperience'])


# In[12]:


data_pred


# In[13]:


import statsmodels.formula.api as smf
model=smf.ols("Salary~YearsExperience",data=salary).fit()
model


# # Model Predictions

# In[14]:


model.predict(data_pred)


# In[15]:


model.predict(salary)


# In[16]:


model.summary()


# In[17]:


pred=model.predict(salary.iloc[:,0])
pred


# In[18]:


model.resid 
model.resid_pearson


# In[19]:


RMSE=np.sqrt(np.mean((np.array(salary['YearsExperience'])-np.array(pred))**2))
RMSE


# In[20]:


plt.scatter(x=salary['YearsExperience'],y=salary['Salary'],color='blue')
plt.plot(salary['YearsExperience'],pred,color='green')
plt.xlabel("YearsExperience")
plt.ylabel("Salary")


# In[21]:


df_new=pd.DataFrame({"YearsExperience":10},index=[1])


# In[22]:


model.predict(df_new)


# ## Delivery_time -> Predict delivery time using sorting time

# In[23]:


time=pd.read_csv("delivery_time.csv")


# In[24]:


time


# # EDA & Data Visualization

# In[25]:


time.shape


# In[26]:


time.head()


# In[27]:


time.info()


# In[28]:


sns.distplot(time['Delivery Time'])


# In[29]:


sns.distplot(time['Sorting Time'])


# # Correlation Analysis

# In[30]:


time.corr()


# In[31]:


time=time.rename({'Delivery Time':'delivery_time','Sorting Time':'sorting_time'},axis=1)


# In[32]:


time


# In[33]:


import statsmodels.formula.api as smf


# In[34]:


sns.regplot(x=time['sorting_time'],y=time['delivery_time'],data=time)
model1=smf.ols("delivery_time~sorting_time",data=time).fit()
model1


# # Model Building

# In[35]:


newdata1=pd.Series([12,13,14,15,16,17])


# In[36]:


data_pred1=pd.DataFrame(newdata1,columns=['sorting_time'])


# In[37]:


data_pred1


# # Model Predictions

# In[38]:


model1.predict(data_pred1)


# In[39]:


model1.predict(time)


# In[40]:


model1.summary()


# In[41]:


pred=model1.predict(time)
pred


# In[42]:


model1.resid
model1.resid_pearson


# In[43]:


RMSE=np.sqrt(np.mean((np.array(time['sorting_time'])-np.array(pred))**2))
RMSE


# In[44]:


model1=smf.ols("delivery_time~np.log(sorting_time)",data=time).fit()


# In[45]:


model1.summary()


# In[46]:


RMSE_log=np.sqrt(np.mean((np.array(time['sorting_time'])-np.array(pred))**2))
RMSE_log


# In[47]:


model2=smf.ols("np.log(delivery_time)~sorting_time",data=time).fit()


# In[48]:


model2.summary()


# In[49]:


RMSE_log=np.sqrt(np.mean((np.array(time['sorting_time'])-np.array(pred))**2))
RMSE_log


# In[50]:


model3=smf.ols("np.log(delivery_time)~np.log(sorting_time)",data=time).fit()


# In[51]:


model3.summary()


# In[52]:


actual=time.delivery_time
pred=model3.predict(time)
residual=actual-pred


# In[53]:


pred


# In[54]:


Newdata=pd.DataFrame({"sorting_time":10.00},index=[1])


# In[55]:


model3.predict(Newdata)


# In[56]:


#Model Deletion Diagnostics
#Detecting Influencers/Outliers
#Cook's Distance

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.formula.api as smf
import numpy as np


# In[57]:


model_influence = model.get_influence()
(c, _) = model_influence.cooks_distance


# In[58]:


#index and value of influencer where c is more than .5
(np.argmax(c),np.max(c))


# In[59]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model)
plt.show()


# In[60]:


k = time.shape[1]
n = time.shape[0]
leverage_cutoff = 3*((k + 1)/n)


# In[61]:


time[time.index.isin([7,18])]


# In[62]:


#See the differences in HP and other variable values
time.head()


# In[63]:


# Improving the model


# In[64]:


#Load the data
time_new=pd.read_csv("delivery_time.csv")


# In[65]:


#Discard the data points which are influencers and reasign the row number (reset_index())
time1=time_new.drop(time_new.index[[7,18]],axis=0).reset_index()


# In[66]:


#Drop the original index
time1=time1.drop(['index'],axis=1)


# In[67]:


time1


# In[68]:


time1=time1.rename({'Delivery Time':'delivery_time','Sorting Time':'sorting_time'},axis=1)


# In[69]:


import statsmodels.formula.api as smf

sns.regplot(x=time1['sorting_time'],y=time1['delivery_time'],data=time1)
model4=smf.ols("delivery_time~sorting_time",data=time1).fit()
model4


# In[70]:


newdata2=pd.Series([12,13,14,15])


# In[71]:


data_pred2=pd.DataFrame(newdata2,columns=['sorting_time'])


# In[72]:


data_pred2


# In[73]:


model4.predict(time1)


# In[ ]:




