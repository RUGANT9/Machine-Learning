# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 12:56:16 2020

@author: ADMIN
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import math

data = pd.read_csv('Placement_Data_Full_Class.csv')
data = data.drop(data.columns[:2], axis = 1)
data = data.drop(data.columns[[1,3]], axis = 1)
data = pd.get_dummies(data, prefix = "En", columns = ['hsc_s'], drop_first = True)
data = pd.get_dummies(data, prefix = "En", columns = ['degree_t'], drop_first = True)
data = pd.get_dummies(data, prefix = "En", columns = ['workex'], drop_first = True)
data = pd.get_dummies(data, prefix = "En", columns = ['specialisation'], drop_first = True)
data = pd.get_dummies(data, prefix = "En", columns = ['status'], drop_first = True)
data['salary'] = data['salary'].fillna(0)
X = data[data.columns[6:]]
Y = data[data.columns[5]]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state = 67)
regres = LinearRegression()
model = regres.fit(X_train, Y_train)
pred = model.predict(X_test)
ax1 = sns.distplot(Y_test, hist = False, color = 'r', label = 'Actual')
sns.distplot(pred, hist = False, color = 'b', label = 'Predicted')