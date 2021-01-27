# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 11:29:57 2020

@author: ADMIN
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Placement_Data_Full_Class.csv')
print(data.head())
data = data.dropna()

for i in range(10):
    print("hello world")
