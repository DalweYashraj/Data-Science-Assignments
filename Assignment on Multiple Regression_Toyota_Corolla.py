#!/usr/bin/env python
# coding: utf-8

# # Assignment on Multiple Regression_Toyota_Corolla

# In[1]:


# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import influence_plot


# In[2]:


# import dataset
toyota=pd.read_csv('ToyotaCorolla.csv',encoding='latin1')
toyota.head()


# # EDA & Visualization

# In[3]:


toyota.info()


# In[4]:


toyota.describe()


# In[5]:


toyota.shape


# In[6]:


toyota1=pd.concat([toyota.iloc[:,2:4],toyota.iloc[:,6:7],toyota.iloc[:,8:9],toyota.iloc[:,12:14],toyota.iloc[:,15:18]],axis=1)
toyota1


# In[7]:


toyota2=toyota1.rename({'Age_08_04':'Age','cc':'CC','Quarterly_Tax':'QT'},axis=1)
toyota2


# In[8]:


# Duplicate Values

toyota2[toyota2.duplicated()]


# In[9]:


toyota3=toyota2.drop_duplicates().reset_index(drop=True)
toyota3


# In[10]:


toyota3.describe()


# # Correlation Analysis

# In[11]:


toyota3.corr()


# In[12]:


sns.set_style(style='darkgrid')
sns.pairplot(toyota3)


# # Model Building 

# In[13]:


model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=toyota3).fit()


# # Model Testing

# In[14]:


model.params


# In[15]:


# Finding tvalues and pvalues
model.tvalues , np.round(model.pvalues,5)


# In[16]:


# Finding rsquared values
model.rsquared , model.rsquared_adj


# # Simple Linear Regression Models

# In[17]:


slr_c=smf.ols('Price~CC',data=toyota3).fit()
slr_c.tvalues , slr_c.pvalues


# In[18]:


slr_d=smf.ols('Price~Doors',data=toyota3).fit()
slr_d.tvalues , slr_d.pvalues


# In[19]:


mlr_cd=smf.ols('Price~CC+Doors',data=toyota3).fit()
mlr_cd.tvalues , mlr_cd.pvalues


# # Calculating VIF

# In[20]:


# Collinearity Problem Check
# Calculate VIF = 1/(1-Rsquare) for all independent variables

rsq_age=smf.ols('Age~KM+HP+CC+Doors+Gears+QT+Weight',data=toyota3).fit().rsquared
vif_age=1/(1-rsq_age)

rsq_KM=smf.ols('KM~Age+HP+CC+Doors+Gears+QT+Weight',data=toyota3).fit().rsquared
vif_KM=1/(1-rsq_KM)

rsq_HP=smf.ols('HP~Age+KM+CC+Doors+Gears+QT+Weight',data=toyota3).fit().rsquared
vif_HP=1/(1-rsq_HP)

rsq_CC=smf.ols('CC~Age+KM+HP+Doors+Gears+QT+Weight',data=toyota3).fit().rsquared
vif_CC=1/(1-rsq_CC)

rsq_DR=smf.ols('Doors~Age+KM+HP+CC+Gears+QT+Weight',data=toyota3).fit().rsquared
vif_DR=1/(1-rsq_DR)

rsq_GR=smf.ols('Gears~Age+KM+HP+CC+Doors+QT+Weight',data=toyota3).fit().rsquared
vif_GR=1/(1-rsq_GR)

rsq_QT=smf.ols('QT~Age+KM+HP+CC+Doors+Gears+Weight',data=toyota3).fit().rsquared
vif_QT=1/(1-rsq_QT)

rsq_WT=smf.ols('Weight~Age+KM+HP+CC+Doors+Gears+QT',data=toyota3).fit().rsquared
vif_WT=1/(1-rsq_WT)

# Putting the values in Dataframe format
d1={'Variables':['Age','KM','HP','CC','Doors','Gears','QT','Weight'],
    'Vif':[vif_age,vif_KM,vif_HP,vif_CC,vif_DR,vif_GR,vif_QT,vif_WT]}
Vif_df=pd.DataFrame(d1)
Vif_df


# In[21]:


# Not an single value of VIF which is greater than 20 (VIF>20),No Collinearity
# So consider all variabes for regression eqn


# # Residual Analysis

# # Test for Normality of Residuals (Q-Q Plot)

# In[22]:


sm.qqplot(model.resid,line='q') # 'q' - A line is fit through the quartiles # line = '45'- to draw the 45-degree diagonal line
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[23]:


list(np.where(model.resid>6000))


# In[24]:


list(np.where(model.resid<-6000))


# # Residual Plot for Homoscedasticity

# In[25]:


def standard_values(vals) : return (vals-vals.mean())/vals.std()


# In[26]:


plt.scatter(standard_values(model.fittedvalues),standard_values(model.resid))
plt.title('Residual Plot')
plt.xlabel('standardized fitted values')
plt.ylabel('standardized residual values')
plt.show() 


# # Residual Vs Regressors

# ### Test for errors or Residuals Vs Regressors or independent 'x' variables or predictors 
# ### using Residual Regression Plots code graphics.plot_regress_exog(model,'x',fig)   
# ### exog = x-variable & endog = y-variable

# In[27]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Age',fig=fig)
plt.show()


# In[28]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'KM',fig=fig)
plt.show()


# In[29]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'HP',fig=fig)
plt.show()


# In[30]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'CC',fig=fig)
plt.show()


# In[31]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Doors',fig=fig)
plt.show()


# In[32]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Gears',fig=fig)
plt.show()


# In[33]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'QT',fig=fig)
plt.show()


# In[34]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'QT',fig=fig)
plt.show()


# # Model Deletion Diagnostics

# ## Detecting Influencers/Outliers

# ## Cook's Distance

# In[35]:


# Get influencers using cook's distance
(c,_)=model.get_influence().cooks_distance
c


# In[36]:


# Plot the influencers using the stem plot
fig=plt.figure(figsize=(20,7))
plt.stem(np.arange(len(toyota3)),np.round(c,3))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')
plt.show()


# In[37]:


np.argmax(c) , np.max(c)


# # High Influence Points

# In[38]:


fig,ax=plt.subplots(figsize=(20,20))
fig=influence_plot(model,ax = ax)


# In[39]:


# Leverage Cuttoff Value = 3*(k+1)/n ; k = no.of features/columns & n = no. of datapoints
k=toyota3.shape[1]
n=toyota3.shape[0]
leverage_cutoff = (3*(k+1))/n
leverage_cutoff


# In[40]:


toyota3[toyota3.index.isin([80])]


# # Improving the Model

# In[41]:


# Creating a copy of data so that original dataset is not affected
toyota_new=toyota3.copy()
toyota_new


# In[42]:


# Eliminate the data points which are influencers and reassign the raw no.
toyota4=toyota_new.drop(toyota_new.index[[80]],axis=0).reset_index(drop=True)
toyota4


# # Build Model

# In[43]:


final_model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=toyota4).fit()
final_model.rsquared,final_model.aic
print("Thus model accuracy is improved to",final_model.rsquared)


# In[44]:


model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=toyota4).fit()
(c,_)=model.get_influence().cooks_distance
c
np.argmax(c) , np.max(c)
toyota4=toyota4.drop(toyota4.index[[np.argmax(c)]],axis=0).reset_index(drop=True)
toyota4


# In[45]:


final_model.rsquared


# In[46]:


final_model.aic


# In[47]:


# Model Predictions

new_data=pd.DataFrame({'Age':15,"KM":50000,"HP":70,"CC":1400,"Doors":5,"Gears":6,"QT":75,"Weight":1020},index=[0])
new_data


# In[48]:


# Manual Prediction of Price
final_model.predict(new_data)


# In[49]:


pred_y=final_model.predict(toyota4)
pred_y

