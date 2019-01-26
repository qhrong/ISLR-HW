# ---Conceptual---
"""
1.
a) Flexible. Since the sample size is large, it’s possible that there are some noise data. A more flexible method can
avoid high influence caused by these noise data, as well as overfitting.
b) Inflexible. Since the sample size is relatively small, a flexible method will be influenced by noise, and given
another random sampling of data, the fit will be significantly different. Therefore, an inflexible method will be less
likely to overfit.
c) Flexible. Given that the relationship is non-linear, an inflexible method cannot capture the non-linearity of the
data, so that an overfit to this certain data may occur.
d) Inflexible. A high variance of error term means that the discrepancy between the values model captured and the real
response values is relatively large, so we don’t want that this noise captured by the model, which will cause overfit.

2.
a) Inference. Because we’re interested in understanding the factors of response variable.
b) Classification. Because the response variable is categorical, success or failure.
c) Prediction. Because we want to use the model to make prediction.

3.
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0.0, 10.0, 0.02)

def squared_bias(x):
    return .002*(-x+10)**3
def variance(x):
    return .002*x**3
def training_error(x):
    return 2.38936 - 0.825077*x + 0.176655*x**2 - 0.0182319*x**3 + 0.00067091*x**4
def test_error(x):
    return 3 - 0.6*x + .06*x**2
def bayes_error(x):
    return x + 1 - x

plt.xkcd()
#frame = plt.gca()
#frame.axes.xaxis.set_ticklabels([])
plt.figure(figsize=(10, 8))
plt.plot(x,squared_bias(x), label='squared bias')
plt.plot(x, variance(x), label='variance')
plt.plot(x, training_error(x), label='training error')
plt.plot(x, test_error(x), label='test error')
plt.plot(x, bayes_error(x), label='Bayes error')
plt.legend(loc='upper center')
plt.xlabel('model flexibility')
plt.show()

"""
4.
a) gender-detection; mnist; package delivery point assignment
b) house price, stock price, revenue level
c) customer clustering, recommending system, social network

5.
Flexible models' advantages: less bias, given enough data we will have better results.
Flexible models' disadvantages: May be overfitting, hard and long to train, less interpretable.

6.
Parametric is the algorithms that make assumptions about the form and our goal is to get the coefficients for the
function by training data; Non-parametric is the algorithms that don’t have strong assumptions on the form of functions.
Parametric algorithm doesn’t need a lot of observations, while non-parametric needs a lot of observations.
Non-parametric doesn't put stricts on the form so it can handle more kinds of data, while parametric can only be
used in the kinds which are consistent to the stricts.
"""


# ---Applied---

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 8
# a)
college = pd.read_csv('/Users/apple/PycharmProjects/ISLR-HW/College.csv')

# b)
college = college.set_index("Unnamed: 0")
college.index.name = 'Name'
# college.drop(college.columns[[0]], axis=1, inplace=True)
print(college.head())

# c)
# produce a numerical summary
college.describe(include='all')
college.describe(include=['number'])

# produce a scatterplot matrix of the first ten columns
sns.set(style="ticks")
sns.pairplot(data=college.iloc[:, 0:10], hue='Private')
# plt.show()
# hue in pairplot: Variable in data to map plot aspects to different colors.

# produce a boxplot
sns.boxplot(x='Outstate', y='Private', data=college)
# plt.show()

# add a new variable
college.loc[college['Top10perc'] > 50, 'Elite'] = 'Yes'
# loc is conditional, iloc needs number of column or row
college['Elite'] = college['Elite'].fillna('No')
college.describe(include=['object'])
sns.boxplot(x='Outstate', y='Elite', data=college)

# produce some histograms
# first step is to set bins
college['PhD'] = pd.cut(college['PhD'], 3, labels=['Low', 'Medium', 'High'])
college['Grad.Rate'] = pd.cut(college['Grad.Rate'], 5, labels=['Very low', 'Low', 'Medium', 'High', 'Very high'])
college['Books'] = pd.cut(college['Books'], 2, labels=['Low', 'High'])
college['Enroll'] = pd.cut(college['Enroll'], 4, labels=['Very low', 'Low', 'High', 'Very high'])

# second step is to make histograms
fig = plt.figure()
plt.subplot(2, 2, 1)  # subplot number 1
college['PhD'].value_counts().plot(kind='bar', title = 'Private')
plt.subplot(2, 2, 2)
college['Grad.Rate'].value_counts().plot(kind='bar', title = 'Grad.Rate')
plt.subplot(2, 2, 3)
college['Books'].value_counts().plot(kind='bar', title = 'Books')
plt.subplot(2, 2, 4)
college['Enroll'].value_counts().plot(kind='bar', title = 'Enroll')
fig.subplots_adjust(hspace=1) # To add space between subplots
# plt.show()

# 9
# a)
auto = pd.read_csv('/Users/apple/PycharmProjects/ISLR-HW/Auto.csv')
auto.info()
# Quantitative predictors: mpg, cylinders, displacement, weight, acceleration, year, origin
# Qualitative predictors: horsepower, name, dtypes

# b)
res = auto.describe()
res.loc['range'] = res.loc['max']-res.loc['min']
print(res.loc['range'])

# c)
print(res.loc[['mean', 'range', 'std']])

# d)
auto_b = auto.drop(auto.index[9:84])
res_b = auto_b.describe()
res_b.loc['range'] = res_b.loc['max']-res_b.loc['min']
print(res_b.loc[['mean', 'range', 'std']])

# e)
# Scatterplot matrix on predictors
sns.set(style="ticks")
sns.pairplot(data=auto)
# plt.show()
"""
Observations:
1- Acceleration seems nearly to a normal distribution. The other variables don't show apparent distribution pattern.
2- Displacement and weight show an apparent positive linear relation.
3- Displacement and weight have a potential negative non-linear relation with mpg.
"""

# f)
"""
Based on the scatterplot matrix:
We can add weight, horsepower and displacement as factors.
"""

# 10
# a)
boston = pd.read_csv('/Users/apple/PycharmProjects/ISLR-HW/Boston.csv')
boston.head()
np.shape(boston)

# b)
# Scatterplot matrix on predictors
sns.set(style="ticks")
sns.pairplot(data=boston)
plt.show()

# Make some scatter plots

plt.scatter(boston['rm'], boston['crim'])
plt.xlabel('RM')
plt.ylabel('CRIM');

plt.scatter(boston['lstat'], boston['crim'])
plt.xlabel('LSAT')
plt.ylabel('CRIM');

plt.scatter(boston['medv'], boston['crim'])
plt.xlabel('MEDV')
plt.ylabel('CRIM');

plt.scatter(boston['tax'], boston['crim'])
plt.xlabel('TAX')
plt.ylabel('CRIM');

# c)
# association with capita crime
boston.corrwith(boston['crim']).sort_values() #sort asc
"""
From the correlation calculated, the predicotrs with the
largest correlations are RAD(index of accessibility to 
radial highways), TAX(full-value property tax rate) and 
LSTAT(percentage of lower status of the population).
"""

# d)
df = pd.DataFrame(boston)
df.sort_values(['crim'], ascending=False).head(5)  # top 5: 380,418,405,410,414
df.sort_values(['tax'], ascending=False).head(5)  # top 5: 492,491,490,489,488
df.sort_values(['ptratio'],ascending=False).head(5)  # top 5: 354,455,135,127,136
res3 = df.describe()
res3.loc['range'] = res3.loc['max']-res3.loc['min']
print(res3['crim'].loc['range'])
print(res3['tax'].loc['range'])
print(res3['ptratio'].loc['range'])

# e)
# number of suburb bound the Charles River
boston['chas'].value_counts()[1] #when it equals to 1

# f)
boston['ptratio'].median()

# g)
# the suburb who has the lowest value of owner occupied homes
boston.loc[boston['age']].min()

# h)
len(boston[boston['rm']>7])  #64
len(boston[boston['rm']>8])  #13
boston[boston['rm']>8].describe()  #CRIM is lower, LSTAT is lower, and INDUS is lower.