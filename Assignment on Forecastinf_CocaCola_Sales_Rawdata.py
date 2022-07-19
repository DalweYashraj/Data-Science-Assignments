#!/usr/bin/env python
# coding: utf-8

# # Assignment on Forecasting_CocaCola_Sales_Rawdata

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Data Driven Methods
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing 


# In[2]:


data=pd.read_csv("CocaCola_Sales_Rawdata.csv")


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


# Segregate Quaeter Values and Year values
data['Quarters'] =0
data['Year'] =0
for i in range(42):
    p = data["Quarter"][i]
    data['Quarters'][i]= p[0:2]
    data['Year'][i]= p[3:5]


# In[7]:


# Getting dummy variables for Quarters Q1, Q2, Q3, Q4 
Quarters_Dummies = pd.DataFrame(pd.get_dummies(data['Quarters']))
data = pd.concat([data,Quarters_Dummies],axis = 1)
data.head()


# In[8]:


# Lineplot for Sales of CocaCola
plt.figure(figsize=(10,7))
plt.plot(data['Sales'], color = 'blue', linewidth=1)


# In[9]:


# Histogram
data['Sales'].hist(figsize=(8,5))


# In[10]:


# Density Plot
data['Sales'].plot(kind = 'kde', figsize=(8,5))


# In[11]:


#boxplot of Quarters Vs. Sales
sns.set(rc={'figure.figsize':(8,5)})
sns.boxplot(x="Quarters",y="Sales",data=data)


# In[12]:


# boxplot of Years Vs. Sales
sns.boxplot(x="Year",y="Sales",data=data)


# In[13]:


from pandas.plotting import lag_plot
lag_plot(data['Sales'])
plt.show()


# In[14]:


plt.figure(figsize=(8,5))
sns.lineplot(x="Year",y="Sales",data=data)


# In[15]:


plt.figure(figsize=(12, 8))
heatmap_y_month = pd.pivot_table(data=data,values="Sales",index="Year",columns="Quarters",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g")


# In[16]:


import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(data.Sales,lags=12)
tsa_plots.plot_pacf(data.Sales,lags=12)
plt.show()


# # Split Train and Test Data

# In[17]:


train = data.head(32)
test = data.tail(10)


# # MA Method

# In[18]:


plt.figure(figsize=(12,4))
data.Sales.plot(label="org")
for i in range(2,8,2):
    data["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc='best')


# In[19]:


# Time Series Decomposition Plot

from statsmodels.tsa.seasonal import seasonal_decompose

decompose_ts_add = seasonal_decompose(data.Sales,period=12)
decompose_ts_add.plot()
plt.show()


# In[20]:


def RMSE(org, pred):
    rmse=np.sqrt(np.mean((np.array(org)-np.array(pred))**2))
    return rmse


# # Simple Exponential Method

# In[21]:


import warnings
warnings.filterwarnings("ignore")

ses_model = SimpleExpSmoothing(train["Sales"]).fit()
pred_ses = ses_model.predict(start = test.index[0],end = test.index[-1])
rmse_ses_model = RMSE(test.Sales, pred_ses)
rmse_ses_model


# # Holt Method

# In[22]:


hw_model = Holt(train["Sales"]).fit()
pred_hw = hw_model.predict(start = test.index[0],end = test.index[-1])
rmse_hw_model = RMSE(test.Sales, pred_hw)
rmse_hw_model


# # Holt Winter Exponential Smoothing With Additive Seasonality and Additive Trend

# In[23]:


hwe_model_add_add = ExponentialSmoothing(train["Sales"],seasonal="add",trend="add",seasonal_periods=4).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = test.index[0],end = test.index[-1])
rmse_hwe_add_add_model = RMSE(test.Sales, pred_hwe_add_add)
rmse_hwe_add_add_model


# # Holts winter exponential smoothing with multiplicative seasonality and additive trend

# In[24]:


hwe_model_mul_add = ExponentialSmoothing(train["Sales"],seasonal="mul",trend="add",seasonal_periods=4).fit() 
pred_hwe_mul_add = hwe_model_mul_add.predict(start = test.index[0],end = test.index[-1])
rmse_hwe_model_mul_add_model = RMSE(test.Sales, pred_hwe_mul_add)
rmse_hwe_model_mul_add_model


# # Model Based Forecasting Methods

# In[25]:


# Data preprocessing for models
data["t"] = np.arange(1,43)
data["t_squared"] = data["t"]*data["t"]

data["log_sales"] = np.log(data["Sales"])

data.head()


# # Linear Model

# In[26]:


import statsmodels.formula.api as smf 

train = data.head(32)
test = data.tail(10)

linear_model = smf.ols('Sales~t',data=train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(test['t'])))
rmse_linear_model = RMSE(test['Sales'],pred_linear)
rmse_linear_model


# # Exponential Model

# In[27]:


Exp = smf.ols('log_sales~t',data=train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(test['t'])))
rmse_Exp_model = RMSE(test['Sales'], np.exp(pred_Exp))
rmse_Exp_model


# # Quadratic Model

# In[28]:


Quad = smf.ols('Sales~t+t_squared',data=train).fit()
pred_Quad = pd.Series(Quad.predict(test[["t","t_squared"]]))
rmse_Quad_model = RMSE(test['Sales'], pred_Quad)
rmse_Quad_model


# # Additive Seasonality Model

# In[29]:


add_sea = smf.ols('Sales~Q1+Q2+Q3',data=train).fit()
pred_add_sea = pd.Series(add_sea.predict(test[['Q1', 'Q2', 'Q3']]))
rmse_add_sea = RMSE(test['Sales'], pred_add_sea)
rmse_add_sea


# # Additive Seasonality Quadratic model

# In[30]:


add_sea_Quad = smf.ols('Sales~t+t_squared+Q1+Q2+Q3',data=train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(test[['Q1','Q2','Q3','t','t_squared']]))
rmse_add_sea_Quad_model = RMSE(test['Sales'], pred_add_sea_quad)
rmse_add_sea_Quad_model 


# # Multiplicative Seasonality model

# In[31]:


Mul_sea = smf.ols('log_sales~Q1+Q2+Q3',data=train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(test))
rmse_Mul_sea = RMSE(test['Sales'], np.exp(pred_Mult_sea))
rmse_Mul_sea


# # Multiplicative Additive Seasonality model

# In[32]:


Mul_Add_sea = smf.ols('log_sales~t+Q1+Q2+Q3',data = train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(test))
rmse_Mul_Add_sea = RMSE(test['Sales'], np.exp(pred_Mult_add_sea))
rmse_Mul_Add_sea


# In[33]:


list = [['Simple Exponential Method',rmse_ses_model], ['Holt method',rmse_hw_model],
          ['HW exp smoothing add',rmse_hwe_add_add_model],['HW exp smoothing mult',rmse_hwe_model_mul_add_model],
          ['Linear Mode',rmse_linear_model],['Exp model',rmse_Exp_model],['Quad model',rmse_Quad_model],
          ['add seasonality',rmse_add_sea],['Quad add seasonality',rmse_add_sea_Quad_model],
          ['Mult Seasonality',rmse_Mul_sea],['Mult add seasonality',rmse_Mul_Add_sea]]


# In[34]:


df = pd.DataFrame(list, columns =['Model', 'RMSE_Value']) 
df


# # From above table ,Quadratic Additive Seasonality model has less RMSE value 
# # So,we can consider this value for model building

# In[35]:


data.head()


# In[36]:


final_model = smf.ols('Sales~t+t_squared+Q1+Q2+Q3',data=data).fit()
pred_final = pd.Series(final_model.predict(data[['Q1','Q2','Q3','t','t_squared']]))
rmse_final_model = RMSE(data['Sales'], pred_final)
rmse_final_model


# In[37]:


# Predict Values

pred_df = pd.DataFrame({'Actual' : data.Sales, 'Predicted' : pred_final})
pred_df


# In[ ]:




