#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:42:44 2019

@author: agatakaczmarek
"""

#import os
#import numpy as np
#import urllib.request
#
#
#temp = np.loadtxt(fname = "Users/agatakaczmarek/Downloads/LATEST/data.txt")




#data = pd.read_csv("/Users/agatakaczmarek/Downloads/LATEST/data.txt", sep=";")

import numpy as np
import pandas as pd

 
#df = pd.read_csv("/Users/agatakaczmarek/Downloads/LATEST/data.txt", sep="\s+|\t+|\s+\t+|\t+\s+", engine= python, index_col = False, header=None)
#df.columns = ["Station-ID", "Series", "Number", "Date", "Temperature(C)", "Uncertainty(C)", "Observations", "Time-of-Observation"]
##%%
#new_header = df.iloc[0]
#print(new_header) #grab the first row for the header
#df = df[1:] #take the data less the header row
#df.columns = new_header #set the header row as the df header


df = pd.read_csv("/Users/agatakaczmarek/Downloads/ai2-science-questions/Elementary-DMC-Dev.csv")

print(df.head())

print(df.tail())

print(type(df))

print(df.shape)

print(df.columns)

print(df.dtypes)

print(df.info())

