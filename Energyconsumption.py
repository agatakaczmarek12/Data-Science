# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astral import Astral
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import linear_model

from sklearn.model_selection import cross_val_score
from sklearn import datasets


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

# Functions to get the sunrise and sunset for a given date
def calcLightHours(date):
    date = datetime.strptime(date, '%Y-%m-%d').date()
    sun = city.sun(date=date, local=True)
    return (sun['sunset'].hour + sun['sunset'].minute / 60) - (sun['sunrise'].hour + sun['sunrise'].minute / 60)

# Make the graphs a bit prettier, and bigger
plt.rcParams['figure.figsize'] = (15, 5)

# I use Astral for sunset and sunrise calculation
# This two lines are used to initialize Astral
a = Astral()
city = a['Madrid']

co = pd.read_csv('/Users/agatakaczmarek/Downloads/Consumptions.csv', sep=';')
tmp = pd.read_csv('/Users/agatakaczmarek/Downloads/Temperatures.csv', sep=';')
tmp2 = pd.read_csv('/Users/agatakaczmarek/Downloads/Temperatures2.csv', sep=';')
nCustomers = pd.read_csv('/Users/agatakaczmarek/Documents/nCustomers.csv', sep=';')

cor = co[['Date', 'Value']].groupby(['Date']).agg("mean").reset_index()
cortmp = pd.merge(cor, tmp, how='outer', on=['Date'])
tmpvalue = cortmp[['Date', 'Value', 'tMean']].groupby(['Date']).agg("mean").reset_index()
plt.scatter(tmpvalue['tMean'], tmpvalue['Value'])

totalhourcons = co.pivot_table(index=("Date", "Hour"), values = "Value", aggfunc ="sum")
totalhourcons.head (100)

cups= co.groupby("Date")["CUPS"].nunique()
cups= pd.DataFrame(cups)

constemp= pd.merge (totalhourcons, tmp, how ="outer", on=["Date"] )

cos = co[['Date', 'Hour', 'Value']].groupby(['Date', 'Hour']).agg("mean").reset_index()
cos.head()

cos['LightHours'] = cos['Date'].map(calcLightHours)


CUPScons= pd.merge(cups,cos, how='outer', on=['Date'])

CUPSconstemp= pd.merge(CUPScons,tmp, how='outer', on=['Date'])

CUPSconstempmea= CUPSconstemp.drop ("tMax", axis=1)

CUPSconstempmean= CUPSconstempmea.drop ("tMin", axis=1)

CUPSconstempmean['Weekday'] = pd.DatetimeIndex(CUPSconstempmean['Date']).weekday

CUPSconstempmean.head()

CUPSconstempmean['Weekend'] = CUPSconstempmean['Weekday'].map(lambda x: 1 if x == 5 or x == 6 else 0)

hourdummies= pd.get_dummies(CUPSconstempmean["Hour"])

hourdummies.columns = hourdummies.columns.astype(str)
hourdummies.dtypes
hourdummies.columns= "Hour" + hourdummies.columns

pd.concat([hourdummies,CUPSconstempmean],axis=1, sort=False)
CUPSconstempmean= pd.merge(CUPSconstempmean, hourdummies, left_index=True, right_index=True)
CUPSconstempmean.head()

CUPSconstempmean= CUPSconstempmean.drop(['Date'], axis=1)


X = CUPSconstempmean.drop(['Value', "Hour"], axis=1)


y= CUPSconstempmean['Value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=25)


#r=linear_model.Ridge(alpha=10, max_iter=1000)

r= svm.SVR(kernel="linear",degree=2, C=100, gamma="auto")

r.fit(X_train, y_train)

y_test = y_test.values.flatten()

y_predicted = r.predict(X_test)

print(np.average(np.abs(y_test - y_predicted) / y_test))

scores = cross_val_score(r, X, y, cv=5)

# Print the accuracy for each fold:
print(scores)

# And the mean accuracy of all 5 folds:
print(scores.mean())

# Prediction dataframe
nCustomers= nCustomers.rename(columns={"datetime": "Date"})

dataset= pd.merge (tmp2, nCustomers, how ="outer", on=["Date"])

dataset= dataset.drop(["tMax", "tMin"], axis=1)

import numpy
from numpy import repeat

hours = numpy.repeat(dataset["Date"],24,axis=None).reset_index()
dataset = pd.merge(dataset,hours, how = "outer" , on = ["Date"])

n=1
for i,row in dataset.iterrows():
    dataset.at[i,"Hour"] = n
    n += 1
    if n == 25:
        n = 1

dataset = dataset[:-24]
#dataset = dataset.drop(["index"])

hourdummies1= pd.get_dummies(dataset["Hour"])

hourdummies1.columns = hourdummies1.columns.astype(str)
hourdummies1.dtypes
hourdummies1.columns= "Hour" + hourdummies1.columns

pd.concat([hourdummies1,dataset],axis=1, sort=False)
dataset= pd.merge(dataset, hourdummies1, left_index=True, right_index=True)

dataset["Weekday"]= pd.DatetimeIndex(dataset["Date"]).weekday

dataset= dataset.drop (["index", "Hour"], axis=1)

dataset['Weekend'] = dataset['Weekday'].map(lambda x: 1 if x == 5 or x == 6 else 0)


#r.fit(X, y)
## new instances where we do not know the answer
#Xnew = dataset
## make a prediction
#ynew = r.predict(Xnew)
## show the inputs and predicted outputs
#
#pred_column = []
#
#for i in range(len(Xnew)):
##    print(ynew[i])
#    pred_column.append(ynew[i])
#
#predictions = dataset
#predictions['Prediction'] = pred_column

#%%

dataset1= dataset.rename(columns={"nCUPS":"CUPS"})
#dataset1 = dataset.drop(["Date"], axis=1)

def calcLightHours(date):
    date = datetime.strptime(date, '%Y-%m-%d').date()
    sun = city.sun(date=date, local=True)
    return (sun['sunset'].hour + sun['sunset'].minute / 60) - (sun['sunrise'].hour + sun['sunrise'].minute / 60)

a = Astral()
city = a['Madrid']
dataset1['LightHours'] = dataset1['Date'].map(calcLightHours)

#%%
dataset1 = dataset1.drop(["Date"], axis=1)
dataset2= dataset1.rename(columns={"nCUPS":"CUPS"})

dataset2 = dataset2[['CUPS','LightHours','tMean','Weekday','Weekend','Hour1.0', 'Hour2.0', 'Hour3.0', 'Hour4.0',
       'Hour5.0', 'Hour6.0', 'Hour7.0', 'Hour8.0', 'Hour9.0', 'Hour10.0',
       'Hour11.0', 'Hour12.0', 'Hour13.0', 'Hour14.0', 'Hour15.0', 'Hour16.0',
       'Hour17.0', 'Hour18.0', 'Hour19.0', 'Hour20.0', 'Hour21.0', 'Hour22.0',
       'Hour23.0', 'Hour24.0']]
#%% Prediction

r.fit(X, y)
# new instances where we do not know the answer
Xnew = dataset2
# make a prediction
ynew = r.predict(Xnew)
# show the inputs and predicted outputs

pred_column = []

for i in range(len(Xnew)):
#    print(ynew[i])
    pred_column.append(ynew[i])

predictions = dataset2
predictions['Prediction'] = pred_column

