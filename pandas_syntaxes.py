# -*- coding: utf-8 -*-

import pandas as pd

# Pandas Structures
# ============================

# 1. Pandas Series: unidimensional dataframe
s = pd.Series([23, 42, 56, 78])
s = pd.Series([23, 24, 25], index=['2019-02-27', '2019-02-27', '2019-02-28'])
# Try to create series from tuple, dict and numpy ndarray
# What happens if I put values of different types in a Series ?

# 2. Pandas DataFrame: Dictionary of Series
football_players = pd.DataFrame({
	'Name': ['Juanfran', 'Godín', 'Diego Costa'],
	'Team': ['Atlético de Madrid', 'Atlético de Madrid', 'Atlético de Madrid'],
	'Age': [34, 33, 30]
})
print(football_players)

# Series and DataFrames relationship
first_row = football_players.loc['Juanfran']
print(type(first_row))
print(first_row)


# Let's play a little

df = pd.read_csv('gapminder.tsv', sep='\t')

print(df.head())

print(df.tail())

print(type(df))

print(df.shape)

print(df.columns)

print(df.dtypes)

print(df.info())

# Filter columns
# =================
dfCountry = df['country']
dfCountry = df.country

df_country_continent_year = df[['country', 'continent', 'year']]

df_continent = df_country_continent_year[[1]]

df_first_last = df[[0, -1]]

some_range = list(range(3, 6))
df_some_columns = df[some_range]

# Filter rows
# =================
print(df.loc[0])
print(df.loc[99])
print(df.loc[df.shape[0] - 1])
print(df.tail(n=1))
print(df.loc[-1]) # will not work

subset_loc = df.loc[0]
subset_head = df.head(n=1)
print(type(subset_loc))
print(type(subset_head))

# These behave the same as loc as index labels are row labels
print(df.iloc[0])
print(df.iloc[99])

# The most general form of indexing
print(df.ix[0])

# Row and column
print(df.ix[42, 'country'])
print(df.loc[42, 'country'])
# Question: how to get the same with iloc ?
# Question: what about df.loc([42, 0]) ?

print(df.ix[[0, 99, 999], ['country', 'lifeExp', 'gdpPercap']])

# Grouping and aggregations
# =============================
print(df.groupby('year')['lifeExp'].mean())
# What if I do df.groupby('year').mean() ?
# What if I do print df.groupby('year')

# Group by two columns
print(df.groupby(['year', 'continent'])[['lifeExp', 'gdpPercap']].mean())

# Columns frequencies
print(df.groupby('continent')['country'].nunique())

# basic plots
life_expectancy = df.groupby(['year'])[['lifeExp']].mean()
life_expectancy.plot()


# Dataframe and Series methods
#=====================================
gdpPercap = df['gdpPercap']
gdpPercap.min() 
# try this
# max, mean, std, append, corr, cov, describe, 
# drop_duplicates, equals, get_values, hist, mode, 
# quantile, replace, sample, sort_values, to_frame, transpose, unique
gdpPercap.quantile()


# Row selection
print(gdpPercap[gdpPercap > gdpPercap.mean()])
# what if I do: print(gdpPercap > gdpPercap.mean()) ?
print(gdpPercap > gdpPercap.mean())

# Vectorized operations
print(gdpPercap * 100)
print(gdpPercap * gdpPercap)

# Things will always align themselves with the index label when actions are performed.
rev_gdp = gdpPercap.sort_index(ascending=False)
print(gdpPercap + rev_gdp)
print(gdpPercap + gdpPercap)

df[gdpPercap > gdpPercap.mean()]

# Dataframe specific
# adding a column
df = football_players
df['NewColumnName'] = [1, 2, 3]
df['NewColumnName'] = [1, 2, 6]
df['NewColumnName'] = [1, 2]

# Playing with dates
df['debut'] = ['2010-10-01', '2011-02-21', '2013-07-03']
# What would I get if I do print(df.dtypes) ?
df['date_born'] = pd.to_datetime(['1990-01-01', '1991-01-01', '1992-01-01'], format='%Y-%m-%d')
df['age_debut'] = (df['Debut_dt'] - df['date_born']).dt.year

# methods for exporting data
# to_csv to_excel to_clipboard tojson to_sql 

# Mergin datasets
# =======================

# dataframe concatenation
s = df['Age']
df = pd.concat([df, df])
df = pd.concat([df, s])
df = pd.concat([df, df], axis=1)
print(df['Name'])

df = df.append(df)
df = df.append(s)

# merging exercises
person = pd.read_csv('survey_person.csv')
survey = pd.read_csv('survey_survey.csv')

# Merging (join) rules - how parameter. default is inner
# left 	Keep all the keys from the left
# right 	Keep all the keys from the right
# outer 	Keep all the keys from both left and right
# inner 	Keep only the keys that exist in the left and right

# experiment with this
pm = person.merge(survey, left_on='ident', right_on='person', how='inner')

# Reshaping your data
df = pd.read_csv('pew_raw.csv')
print(df)
# melt function
print(pd.melt(df, id_vars='religion', var_name='income', value_name='count'))

# When you want to use scikitlearm
# (With the exception of HotEncoding and ColumnTransformer)
m = df.values

# NaN values
from numpy import NaN, NAN, nan

df = pd.read_csv('country_timeseries.csv')
df.count()
df.fillna(0)
df.fillna(method='ffill')
df.fillna(method='bfill')
df.interpolate()
df.dropna() # look at parameter how in documentation

# Create a new column with some calculation of other columns
# What happens to NaN values ?



# Plotting
#====================================
# 1. matplotlib
# https://matplotlib.org/
import matplotlib.pyplot as plt

# 2. seaborn
# https://seaborn.pydata.org/

import seaborn as sns
anscombe = sns.load_dataset("anscombe")
print(anscombe)

# Matplotlib
# ====================================
ds1 = anscombe[anscombe['dataset'] == 'I']
plt.plot(ds1['x'], ds1['y'])
plt.plot(ds1['x'], ds1['y'], 'o')

# A single plot is easy. Let's do something more interesting
ds1 = anscombe[anscombe['dataset'] == 'I']
ds2 = anscombe[anscombe['dataset'] == 'II']
ds3 = anscombe[anscombe['dataset'] == 'III']
ds4 = anscombe[anscombe['dataset'] == 'IV']

fig = pit.figure()
axes1 = fig.add_subplot(2 , 2, 1)
axes2 = fig.add_subplot(2 , 2, 2)
axes3 = fig.add_subplot(2 , 2, 3)
axes4 = fig.add_subplot(2 , 2, 4)

axes1.set_title("dataset I")
axes2.set_title("dataset II")
axes3.set_title("dataset III")
axes4.set_title("dataset IV")
fig.suptitle("Anscombe Data")

axesl.plot(ds1['x'], ds1['y'], 'o')
axes2.plot(ds2['x'], ds2['y'], 'o')
axes3.plot(ds3['x'], ds3['y'], 'o')
axes4.plot(ds4['x'], ds4['y'], 'o')

# Terminology: axes and axis

# Some plots examples
tips = sns.load_dataset("tips")
print(tips.head())

# univariate plot
fig = pit.figure()
axes1 = fig.add_subplot(1, 1, 1)
axes1.hist(tips['total_bill'], bins=10)
axes1.set_title('Histogram of Total Bill')
axes1.set_xlabel('Frequency' )
axes1.set_ylabel('Total Bill')
fig.show ()

# bivariate plots
scatter_plot = plt.figure()
axes1 = scatter_plot.add_subplot(1, 1, 1)
axes1.scatter(tips['total_bill'], tips['tip'])
axes1.set_title('Scatterplot of Total Bill vs Tip')
axes1.set_xlabel('Total Bill')
axes1.set_ylabel('Tip') scatter_plot.show()

boxplot = pit.figure()
axes1 = boxplot.add_subplot(1, 1, 1)
axes1.boxplot([tips[tips['sex'] == 'Female']['tip'], tips [tips ['sex'] == 'Male']['tip']], labels=['Female', 'Male'])
axes1.set_xlabel('Sex')
axes1.set_ylabel('Tip')
axes1.set_title('Boxplot of Tips by Sex')

# multivariate plot
def recode_sex(sex):
	if sex == 'Female':
		return 0
	else:
		return 1

tips['sex_color'] = tips['sex'].apply(recode_sex)

scatter_plot = plt.figure()
axes1 = scatter_plot.add_subplot(1, 1, 1)
axes1.scatter(x=tips['total_bill'], y=tips['tip'], s=tips['size'] * 10, tips['sex_color'])
axes1.set_title('Total Bill vs Tip colored by Sex and sized by Size'
axes1.set_xlabel('Total Bill')
axes1.set_ylabel('Tip')
scatter_plot.show()

# Seaborn
# ==================================
hist = sns.distplot(tips['total_bill'])
hist.set_title('Total Bill Histogram with Density Plot')

scatter = sns.regplot(x='total_bill', y='tip', data=tips)
scatter.set_title('Scatterplot of Total Bill and Tip')
scatter.set_xlabel('Total Bill')
scatter.set_ylabel('Tip')

scatter = sns.jointplot(x='total_bill', y='tip', data=tips)
scatter.set_axis_labels(xlabel='Total Bill', ylabel='Tip')
scatter.fig.suptitle('Joint plot of Total Bill and Tip', fontsize=20, y=1.03)

sns.pairplot(tips)
sns.pairplot(tips, hue='sex')

# Note that we can use sunborn or matplotlib functions in map
facet = sns.FacetGrid(tips, col='time')
facet.map(sns.distplot, 'total_bill')

facet = sns.FacetGrid(tips, col='day', hue='sex')
facet = facet.map(plt.scatter, 'total_bill', 'tip')
facet = facet.add_legend()
