# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 12:18:12 2020

@author: ADMIN
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('train.csv')
train_data = train_data.drop('Ticket', axis = 1)
train_data_target = train_data.iloc[:, 1] 
train_data = train_data.drop('Survived', axis = 1)
train_data = train_data.drop(['Name', 'Cabin', 'Embarked', 'Fare', 'PassengerId'], axis = 1)
train_data = train_data.fillna(30)
test_data = pd.read_csv('test.csv')
ans_data = pd.DataFrame(test_data['PassengerId'])
test_data = test_data.drop(['Name', 'Cabin', 'Embarked', 'Fare', 'PassengerId'], axis = 1)
test_data = test_data.fillna(30)
test_data = test_data.drop('Ticket', axis = 1)
train_data = pd.get_dummies(train_data, drop_first = True)
test_data = pd.get_dummies(test_data, drop_first = True)
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(train_data, train_data_target)
pred = lr_model.predict(test_data)
ans_data['Survived'] = pred
ans_data.to_csv('Submit.csv', index = False)
age_data = pd.DataFrame(train_data['Age'])
age_data = age_data.sort_values(by='Age', ascending = True)
age_data['Survived'] = train_data_target