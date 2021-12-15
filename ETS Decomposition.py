#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math
import pandas_gbq
from pylab import rcParams
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt


import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

def data_cleaner_new(df):
    df['Transaction_date']=pd.to_datetime(df['Transaction_date'])
    df=df.groupby('Transaction_date')[['QTY']].sum().reset_index()
    df = df.set_index('Transaction_date')
    df=df[['QTY']].resample('M').sum()
    df.rename(columns = {'QTY':'Quantity'}, inplace = True) 
#     df['Transaction_date']=df.index
#     df.reset_index(level=0, inplace=True)
    return df


def Holt_winter_output_generator(df):
    #Splitting data into test and train sets, without shuffling
    df2=df    
    df2['Transaction_date']=df2.index
#     x_train,x_test,y_train,y_test=train_test_split(df['Transaction_date'],df[['Amount','Quantity']],test_size=0.1,random_state=123,shuffle=False)
    df_train=df2[df2['Transaction_date']<='2020-11-30']
    df_test=df2[df2['Transaction_date']>'2020-11-30']
    
    #      Ideal Seasonal Period determination using least RMSE for Quantity
    rmse_dict_2={}
    for i in range(2,14):
        fit2 = ExponentialSmoothing(df_train['Quantity'] ,seasonal_periods=i ,trend='add', seasonal='add', damped=True).fit(use_boxcox=True)
        df_test['Predicted_Quantity'] = fit2.forecast(len(df_test['Quantity']))
        df_test['Predicted_Quantity']=df_test['Predicted_Quantity'].fillna(0)
        rmse_dict_2[i]=math.sqrt(mean_squared_error(list(df_test['Quantity']),list(df_test['Predicted_Quantity'])))
    temp2 = min(rmse_dict_2.values()) 
    res2 = [key for key in rmse_dict_2 if rmse_dict_2[key] == temp2] 
    res2=res2[0]
    
    #     #Prediction for Quantity
    
    fit2= ExponentialSmoothing(np.asarray(df_train['Quantity']) ,seasonal_periods=res2 ,trend='add', seasonal='add', damped=True).fit(use_boxcox=True)
    df_test['Predicted_Quantity'] = fit2.forecast(len(df_test))
    df_test['Diff in Qty']=abs(df_test['Predicted_Quantity']-df_test['Quantity'])
    df_test['Diff % in Qty']=(df_test['Diff in Qty'])*100/(df_test['Quantity'])
    df_test.drop(df_test.tail(1).index,inplace=True)
    
    print ('The average % error in prediction is : {} %'.format(round(df_test['Diff % in Qty'].mean()),4))
    return  df_test

def Holt_winter_mul_trend(df):
    #Splitting data into test and train sets, without shuffling
    df2=df    
    df2['Transaction_date']=df2.index
#     x_train,x_test,y_train,y_test=train_test_split(df['Transaction_date'],df[['Amount','Quantity']],test_size=0.1,random_state=123,shuffle=False)
    df_train=df2[df2['Transaction_date']<='2020-11-30']
    df_test=df2[df2['Transaction_date']>'2020-11-30']
    
    #      Ideal Seasonal Period determination using least RMSE for Quantity
    rmse_dict_2={}
    for i in range(2,14):
        fit2 = ExponentialSmoothing(df_train['Quantity'] ,seasonal_periods=i ,trend='mul', seasonal='add', damped=True).fit(use_boxcox=True)
        df_test['Predicted_Quantity'] = fit2.forecast(len(df_test['Quantity']))
        df_test['Predicted_Quantity']=df_test['Predicted_Quantity'].fillna(0)
        rmse_dict_2[i]=math.sqrt(mean_squared_error(list(df_test['Quantity']),list(df_test['Predicted_Quantity'])))
    temp2 = min(rmse_dict_2.values()) 
    res2 = [key for key in rmse_dict_2 if rmse_dict_2[key] == temp2] 
    res2=res2[0]
    
    #     #Prediction for Quantity
    
    fit2= ExponentialSmoothing(np.asarray(df_train['Quantity']),seasonal_periods=res2 ,trend='mul', seasonal='add', damped=True).fit(use_boxcox=True)
    df_test['Predicted_Quantity'] = fit2.forecast(len(df_test))
    df_test['Diff in Qty']=abs(df_test['Predicted_Quantity']-df_test['Quantity'])
    df_test['Diff % in Qty']=(df_test['Diff in Qty'])*100/(df_test['Quantity'])
    df_test.drop(df_test.tail(1).index,inplace=True)
    print ('The average % error in prediction is : {} %'.format(round(df_test['Diff % in Qty'].mean()),4))
    
    return  df_test

def Holt_winter_no_damp(df):
    #Splitting data into test and train sets, without shuffling
    df2=df    
    df2['Transaction_date']=df2.index
#     x_train,x_test,y_train,y_test=train_test_split(df['Transaction_date'],df[['Amount','Quantity']],test_size=0.1,random_state=123,shuffle=False)
    df_train=df2[df2['Transaction_date']<='2020-11-30']
    df_test=df2[df2['Transaction_date']>'2020-11-30']
    
    #      Ideal Seasonal Period determination using least RMSE for Quantity
    rmse_dict_2={}
    for i in range(2,14):
        fit2 = ExponentialSmoothing(df_train['Quantity'] ,seasonal_periods=i ,trend='add', seasonal='add').fit(use_boxcox=True)
        df_test['Predicted_Quantity'] = fit2.forecast(len(df_test['Quantity']))
        df_test['Predicted_Quantity']=df_test['Predicted_Quantity'].fillna(0)
        rmse_dict_2[i]=math.sqrt(mean_squared_error(list(df_test['Quantity']),list(df_test['Predicted_Quantity'])))
    temp2 = min(rmse_dict_2.values()) 
    res2 = [key for key in rmse_dict_2 if rmse_dict_2[key] == temp2] 
    res2=res2[0]
    
    #     #Prediction for Quantity
    
    fit2= ExponentialSmoothing(np.asarray(df_train['Quantity']) ,seasonal_periods=res2 ,trend='add', seasonal='add').fit(use_boxcox=True)
    df_test['Predicted_Quantity'] = fit2.forecast(len(df_test))
    df_test['Diff in Qty']=abs(df_test['Predicted_Quantity']-df_test['Quantity'])
    df_test['Diff % in Qty']=(df_test['Diff in Qty'])*100/(df_test['Quantity'])
    df_test.drop(df_test.tail(1).index,inplace=True)
    print ('The average % error in prediction is : {} %'.format(round(df_test['Diff % in Qty'].mean()),4))
    
    return  df_test

def Decomposition_plot(cleaned_df):
    
    rcParams['figure.figsize'] = 18, 8
    decomposition = sm.tsa.seasonal_decompose(cleaned_df, model='additive')
    fig = decomposition.plot()
    plt.show()
    

def Graphical_validation(df):
    #Splitting data into test and train sets, without shuffling
    
    df2=df    
    df2['Transaction_date']=df2.index
#     x_train,x_test,y_train,y_test=train_test_split(df['Transaction_date'],df[['Amount','Quantity']],test_size=0.1,random_state=123,shuffle=False)
    df_train=df2[df2['Transaction_date']<='2020-11-30']
    df_test=df2[df2['Transaction_date']>'2020-11-30']
    
    #      Ideal Seasonal Period determination using least RMSE for Quantity
    rmse_dict_2={}
    for i in range(2,14):
        fit2 = ExponentialSmoothing(df_train['Quantity'] ,seasonal_periods=i ,trend='add', seasonal='add', damped=True).fit(use_boxcox=True)
        df_test['Predicted_Quantity'] = fit2.forecast(len(df_test['Quantity']))
        df_test['Predicted_Quantity']=df_test['Predicted_Quantity'].fillna(0)
        rmse_dict_2[i]=math.sqrt(mean_squared_error(list(df_test['Quantity']),list(df_test['Predicted_Quantity'])))
    temp2 = min(rmse_dict_2.values()) 
    res2 = [key for key in rmse_dict_2 if rmse_dict_2[key] == temp2] 
    res2=res2[0]
    
    #Graph Plot
    
    fit2= ExponentialSmoothing(np.asarray(df_train['Quantity']) ,seasonal_periods=res2 ,trend='add', seasonal='add', damped=True).fit(use_boxcox=True)
    df_test['Holt_Winter'] = fit2.forecast(len(df_test))
    plt.figure(figsize=(16,8))
    plt.plot(df_train['Quantity'], label='Train')
    plt.plot(df_test['Quantity'], label='Test')
    plt.plot(df_test['Holt_Winter'], label='Predicted')
    plt.legend(loc='best')
    plt.show()
    
    print('RMSE: ',temp2)


# In[ ]:


Decomposition_plot(data_cleaner_new(tx_data_IN))

Holt_winter_output_generator(data_cleaner_new(tx_data_IN))

Holt_winter_no_damp(data_cleaner_new(tx_data_IN))


# In[ ]:




