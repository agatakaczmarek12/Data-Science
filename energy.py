# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
###############################################################################
# Power consumption dataset
###############################################################################
plt.rcParams['figure.figsize'] = (15, 5)
# Read the data
co= pd.read_csv('/Users/agatakaczmarek/Downloads/Consumptions.csv', sep=';')
tmp= pd.read_csv('/Users/agatakaczmarek/Downloads/Temperatures.csv', sep=';')

# Some simple statistics
co.describe()
tmp.describe()
#print co.head
#ankdf = pd.read_csv('bank.csv',sep=';') # check the csv file before to know that 'comma' here is ';'
#print bankdf.head(3)
#print list(bankdf.columns)# show the features and label 
#print bankdf.shape # instances vs features + label (4521, 17)

# Rate possible values
co['Rate'].unique()

# Add three useful columns
co['Year'] = pd.DatetimeIndex(co['Date']).year
co['Month'] = pd.DatetimeIndex(co['Date']).month
co['Weekday'] = pd.DatetimeIndex(co['Date']).weekday

# Try some commands to understanda your data
co[['Date', 'Value']].groupby('Date').agg(sum).plot()
co[['Date', 'Value']].groupby('Date').agg("mean").plot()

co[['Month', 'Value']].groupby('Month').agg("mean").plot()
co[['Weekday', 'Value']].groupby('Weekday').agg("mean").plot()
co[['Hour', 'Value']].groupby('Hour').agg("mean").plot()

outliers = co[co['Value'] == 0]
outliers = co[co['Value'] > 5000]

# What is the impact of rate?
co[['Rate', 'Value']].groupby('Rate').agg("mean")
co21a = co[co['Rate'] == '2.1A']
co21a[['Hour', 'Value']].groupby('Hour').agg("mean").plot()

cor = co[['Date', 'Value']].groupby(['Date']).agg("mean").reset_index()
cortmp = pd.merge(cor, tmp, how='outer', on=['Date'])
tmpvalue = cortmp[['Date', 'Value', 'tMean']].groupby(['Date']).agg("mean").reset_index()
plt.scatter(tmpvalue['tMean'], tmpvalue['Value'])




