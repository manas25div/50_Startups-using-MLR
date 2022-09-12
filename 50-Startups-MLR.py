# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 13:10:33 2022

@author: MANAS
"""
# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. Read/Importing the dataset
dataset=pd.read_csv(r'F:\DS\Workshop\50_Startups.csv')

# 2.Splitting the data Y (dependent -Profit- Last column) X ( Independent data - Rest all columns)
X=dataset.iloc[:,:-1] # All rows, all columns except last
y=dataset.iloc[:,4] # All rows but only Column no. 4

# 3.LabelEncoding for State column- Add 3 columns with each state name
X=pd.get_dummies(X) # Now X have 6 columns


# 4.For Linear Model Base
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=0)

# All 50 rows (40 - Train, 10- Test)

# 5. Model Building
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# 6.Data X_test Fit to Model for y Prediction
y_pred=regressor.predict(X_test)

# 7 Backword Elimination
# In X we have 3 attributes, so to choose,which one we use from our model, we use this technique

import statsmodels.formula.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1) # Add column constant 1 for all 50 rows

# 8  Optimization
import statsmodels.api as sm
X_opt =X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()































