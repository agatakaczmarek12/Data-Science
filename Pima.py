#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 17:32:34 2019

@author: agatakaczmarek
"""
#%%
# Always use this header

##%matplotlib inline
#import pandas as pd
#import matplotlib.pyplot as plt
#import numpy as np

# Make the graphs a bit prettier, and bigger
#pd.set_option('display.mpl_style', 'default')
#plt.rcParams['figure.figsize'] = (15, 5)

################################################################################
## Toy dataset statistics
################################################################################
#data = pd.read_csv("/Users/agatakaczmarek/Desktop/pima-indians-diabetes.data.csv ", sep=";")
#data.hist()
#data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
#data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
#plt.matshow(data.corr())
#
#from pandas.plotting import scatter_matrix
#scatter_matrix(data)

#%%

#%%
# Always use this header

#%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Make the graphs a bit prettier, and bigger
#pd.set_option('display.mpl_style', 'default')
plt.rcParams['figure.figsize'] = (15, 5)

###############################################################################
# Toy dataset statistics
###############################################################################
data = pd.read_csv("/Users/agatakaczmarek/Desktop/pima-indians-diabetes.data.csv", sep=";")
data.hist()
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
plt.matshow(data.corr())

from pandas.plotting import scatter_matrix
scatter_matrix(data)