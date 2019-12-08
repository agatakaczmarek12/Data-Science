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

import seaborn as sns
plt.rcParams['figure.figsize'] = (15, 5)

hourly = pd.read_csv("/Users/agatakaczmarek/Downloads/Bike-Sharing-Dataset/hour.csv", sep=',')

hourly.describe()

sns.heatmap(hourly.corr())

hourly[['atemp', 'cnt']].groupby('cnt').agg("mean").plot ()

plt.rcParams['figure.figsize'] = (15, 5)
