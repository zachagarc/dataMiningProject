#!/usr/bin/env python
# coding: utf-8

# In[17]:


# Ignore warnings for readability (will fix warnings)
import warnings

# Standard Pandas and Numpy imports
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')


# Import breast cancer dataset
# Used utf-8-sig to remove import error
data = pd.read_csv('breast-cancer-wisconsin.data', delimiter = ',',
                   encoding = 'utf-8-sig', header = None, 
                   names = ['id', 'Clump Thickness', 'Uniformity of Cell Size',
                            'Uniformity of Cell Shape', 'Marginal Adhesion',
                            'Single Epithelial Cell Size', 'Bare Nuclei', 
                            'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 
                            'Class'])
data.head()

# Check if data is imbalanced. Here, the data is split 34% to 66% (reasonably imbalanced)
data['Class'].value_counts()

# Replace ? values with nan
data.replace('?', np.nan, inplace = True)
data.head()

#Remove outliers based on IQR (Inter quartile range)
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]
data.shape

# Seperate features and labels
X = data[['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli']]
y = data['Class']
# Check for null objects. Here, only Bare Nuclei has null objects
X.info()
# 16 null values that need ot be replaced
X['Bare Nuclei'].isna().sum()
# Replace NaN values with most frequent value
X['Bare Nuclei'].fillna(inplace = True, value = 1)

# Confirm the null values were replaced
X.isna().sum()
# Import Normalizer from sklearn
from sklearn.preprocessing import Normalizer
nm = Normalizer(copy = True)

# Normalize all rows and columns
X.loc[:, :] = nm.fit_transform(X.loc[:, :])
X.head()

# Import KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

# Initialize KNN params that will be tested
kOptions = range(1, 31)
algorithmOptions = ['ball_tree', 'kd_tree', 'brute']
gridParamsKNN = dict(n_neighbors = kOptions, algorithm = algorithmOptions)
# Import grid search cross validation
from sklearn.model_selection import GridSearchCV

# Initialize the grid search with KNN classifier using 10 - cross fold validation
# The scoring method was set to recall since we are interested in how well it detects anamolies with insufficient data
gridKNN = GridSearchCV(knn, gridParamsKNN, 'recall_macro', cv = 10)
gridKNN.fit(X, y)

# RESULTS
CVScoreKNN = gridKNN.best_score_
print('10 fold cross validated recall score for KNN classifier: ', CVScoreKNN)
# Import random forest classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

# Initialize RFC params that will be tested
critOptions = ['gini', 'entropy']
gridParamsRFC = dict(criterion = critOptions)
# Initialize the grid search with RFC classifier using 10 - cross fold validation
# The scoring method was set to recall since we are interested in how well it detects anomalies with insufficient data
gridRFC = GridSearchCV(rfc, gridParamsRFC, 'recall_macro', cv = 10)
gridRFC.fit(X, y)

#RESULTS
CVScoreRFC = gridRFC.best_score_
print('10 fold cross validated recall score for RF classifier: ', CVScoreRFC)
# Import support vector machine classifier 
from sklearn.svm import SVC
svc = SVC()

# Initialize SVC params that will be tested
cOptions = [.03, .01, .1, .5, .9, 2, 5, 10]
kernelOptions = ['linear', 'rbf', 'sigmoid']
gridParamsSVC = dict(C = cOptions, kernel = kernelOptions)
# Initialize the grid search with SVM classifier using 10 - cross fold validation
# The scoring method was set to recall since we are interested in how well it detects anomalies with insufficient data
gridSVC = GridSearchCV(svc, gridParamsSVC, 'recall_macro', cv = 10)
gridSVC.fit(X, y)

#RESULTS
CVScoreSVC = gridSVC.best_score_
print('10 fold cross validated recall score for SVM classifier: ', CVScoreSVC)
# Import naive bayes classifier and coss validation score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

nbc = GaussianNB()
CVScoreNBC = cross_val_score(nbc, X, y, cv = 10, scoring = 'recall_macro').mean()

#RESULTS
print('10 fold cross validated recall score for NB classifier: ', CVScoreNBC)
print('10 fold cross validated recall score for KNN classifier: ', CVScoreKNN)
print('10 fold cross validated recall score for RF classifier: ', CVScoreRFC)
print('10 fold cross validated recall score for SVM classifier: ', CVScoreSVC)
print('10 fold cross validated recall score for NB classifier: ', CVScoreNBC)


# In[ ]:




