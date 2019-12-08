# -*- coding: utf-8 -*-
#%%

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import VarianceThreshold

from sklearn.pipeline import Pipeline

from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

from sklearn.pipeline import make_pipeline

plt.rcParams['figure.figsize'] = (15, 5)

df = pd.read_csv ("/Users/agatakaczmarek/Documents/train (3).csv", index_col='Id')

""" 
Know yor data
"""

#print(df.describe())
print(df.info())
#print(df.columns[df.isnull().any()])

#for column in df:
#    print(column, df[column].unique())

#columns with null values
#df.columns[df.isnull().any()]

# null values weight
#n = df.isnull().sum()/len(df)
#n = n[n > 0]
#n.sort_values(inplace=True)
#n
    
# Distribution of output
# sns.distplot(df['SalePrice'])

# Correlation of numeric variables
#corr = df.select_dtypes(include=[np.number]).corr()
#print(corr['SalePrice'].sort_values(ascending=False))

# Plotting pairs of variables
#sns.jointplot(x=df['OverallQual'], y=df['SalePrice'])
#sns.jointplot(x=df['GrLivArea'], y=df['SalePrice'])

# Description of categorical variables
df.select_dtypes(exclude=[np.number]).describe()

"""
Simplest possible cleaning
"""
def simplest_cleaning(model, cvStrategy, scoring):
    #X = df[['OverallQual', 'GrLivArea', 'GarageCars']]
    X = df.drop('SalePrice', 1)
    
    # Select numeric columns
    X = X.select_dtypes(include=[np.number])
    
    # Select not null columns
    X = X[X.columns[X.notnull().all()]]
    
    y = df['SalePrice']
    
    scores = cross_val_score(model, X, y.values.flatten(), cv=cvStrategy, scoring=scoring)
    print(np.average(scores))

"""
Imputing null values
"""
def impute_null_values(model, cvStrategy, scoring):
    X = df.drop('SalePrice', 1)
    
    # Select numeric columns
    X = X.select_dtypes(include=[np.number])
    
    # Impute null values 
#    imp = SimpleImputer(missing_values=np.NaN, strategy='median')|
    imp.fit(X)
    X = pd.DataFrame(imp.transform(X))
       
    y = df['SalePrice']
    
    scores = cross_val_score(model, X, y.values.flatten(), cv=cvStrategy, scoring=scoring)
    print(np.average(scores))

"""
Encode categorical values
"""
def encode_categorical_values(model, cvStrategy, scoring):
    X = df.drop('SalePrice', 1)
    
    si = ('si', SimpleImputer(strategy='constant', fill_value='MISSING'))
    ohe = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))
    pipec = Pipeline([si, ohe])
#    
#    Xc = X.select_dtypes(exclude=[np.number])
#    Xct = pd.DataFrame(pipec.fit_transform(Xc))
# 
#    imp = ('imp', SimpleImputer(missing_values=np.NaN, strategy='median'))
#    scl = ('scl', StandardScaler())
#    pipen = Pipeline([imp, scl])
    
    Xn = X.select_dtypes(include=[np.number])
    Xnt = pd.DataFrame(pipen.fit_transform(Xn))
    
    X = pd.concat([Xct, Xnt], axis=1, sort=False)
    y = df['SalePrice']
    
    scores = cross_val_score(model, X, y.values.flatten(), cv=cvStrategy, scoring=scoring)
    print(np.average(scores))
    
def more_clever_cleaning(model, cvStrategy, scoring):
    
    newdf = df[df['SalePrice'] / df['GrLivArea'] > 40]
    newdf = newdf.loc[:, newdf.isnull().mean() < .25]

    X = newdf.drop('SalePrice', 1)

    y = newdf['SalePrice']
    
    si = SimpleImputer(strategy='constant', fill_value='MISSING')
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    
    Xc = X.select_dtypes(exclude=[np.number])
    Xct = pd.DataFrame(si.fit_transform(Xc))
    Xct = pd.DataFrame(ohe.fit_transform(Xct))
 
    imp = SimpleImputer(missing_values=np.NaN, strategy='median')
    scl = StandardScaler()
    
    Xn = X.select_dtypes(include=[np.number])
    Xnt = pd.DataFrame(imp.fit_transform(Xn))
    Xnt = pd.DataFrame(scl.fit_transform(Xnt))
    
    X = pd.concat([Xct, Xnt], axis=1, sort=False)
    
    """
    lof = LocalOutlierFactor(contamination=0.1)
    o = lof.fit_predict(X)
    Xo = pd.DataFrame()
    yo = pd.DataFrame()
    for i in range(0, o.shape[0]):
        if o[i] == 1:
            Xo = Xo.append(X.iloc[[i]])
            yo = yo.append(pd.DataFrame(y.iloc[[i]]))
    scores = cross_val_score(model, Xo, yo.values.flatten(), cv=cvStrategy, scoring=scoring)
    """
    scores = cross_val_score(model, X, y.values.flatten(), cv=cvStrategy, scoring=scoring)
    print(np.average(scores))

"""
All Together
"""
rs = 0
models = [
    ['Linear', linear_model.Ridge(alpha=10, max_iter=1000)],
    ['Random Forest', RandomForestRegressor(n_estimators=100, n_jobs=-1)],
    ['Support Vector Machine', svm.SVR(kernel='linear', C=1000)]
]

cvStrategies = [
    ['Suffle Split', ShuffleSplit(n_splits=10, test_size=0.3, random_state=rs)],
    ['K-Fold', KFold(n_splits=5, shuffle=True, random_state=rs)]
]

cvStrategy = cvStrategies[1]
scoring = 'r2'
#scoring = 'neg_mean_squared_log_error'
for model in models:
    print()
    print(cvStrategy[0] + ' - ' + model[0])
    #if model[0] != 'Support Vector Machine':
    #    simplest_cleaning(model[1], cvStrategy[1], scoring)
    #    impute_null_values(model[1], cvStrategy[1], scoring)
    encode_categorical_values(model[1], cvStrategy[1], scoring)
    more_clever_cleaning(model[1], cvStrategy[1], scoring)

