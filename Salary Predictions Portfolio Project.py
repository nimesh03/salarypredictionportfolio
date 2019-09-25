#!/usr/bin/env python
# coding: utf-8

# # Salary Predictions Based on Job Descriptions

# # Part 1 - DEFINE

# ### ---- 1 Define the problem ----

# The problem we are trying to solve is to predict job salaries for a set of new job postings, based on a trained model that uses current job posting and salary information.

# In[120]:


#my info
__author__ = "Nimesh Yogarajan"
__email__ = "nimesh19@hotmail.ca"

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import shuffle
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


# In[57]:


#Repetitive Functions 

#Loading files
def loadFile(file):
    df = pd.read_csv(file)
    return df
    
#Cleaning data (remove duplicate Job IDs and Salaries that are 0)
def cleanData(df):
    cleanData = df.drop_duplicates(subset='jobId')
    cleanData = df[df.salary > 0]
    return cleanData

#Feature to encode all categorical variables and combines result with continuous variables 
def one_hot_encoding(df, cat_vars=None, num_vars=None):
    cat_df = pd.get_dummies(df[cat_vars])
    num_df = df[num_vars].apply(pd.to_numeric)
    return pd.concat([cat_df, num_df], axis=1)

#returning target df
def getTarget(df, target):
    return df[target]

def plot_feat(df, column):
    
    plt.figure(figsize = (13,6))
    plt.subplot(1,2,1)
    
    if df[column].dtype == 'int64':
        df[column].value_counts().sort_index().plot()
    else:
        #change categorical vairanle to category and order level by mean salary in each category
        mean = df.groupby(column)['salary'].mean()
        df[column] = df[column].astype('category')
        levels = mean.sort_values().index.tolist()
        df[column].cat.reorder_categories(levels, inplace = True)
        df[column].value_counts().plot()
    plt.xticks(rotation=45)
    plt.xlabel(column)
    plt.ylabel("Counts")
    plt.subplot(1,2,2)
    
    if df[column].dtype == 'int64' or column == 'companyId':
        #plot mean salary, fill between (mean - std, mean + std)
        mean = df.groupby(column)['salary'].mean()
        std = df.groupby(column)['salary'].std()
        mean.plot()
        plt.fill_between(range(len(std.index)), mean.values-std.values, mean.values + std.values,                          alpha = 0.1)
    
    else:
        sb.boxplot(x=column, y='salary', data=joined_df)
        
    plt.xticks(rotation=45)
    plt.ylabel('Salaries')
    plt.show()

#scatter plots 
def scatter_plot(df, col):
    plt.figure(figsize = (14,6))
    plt.scatter(df[col], df['salary'] )
    plt.ylabel('salary')
    plt.xlabel(col)

#regression plots
def reg_plot(df, col):
    plt.figure(figsize=(14,6))
    sb.regplot(x=df[col], y = df['salary'], data = df, line_kws = {'color' : 'red'})    
    plt.ylim(0,)
    
#residual plots 
def res_plot(df,col):
    plt.figure(figsize=(14,6))
    sb.residplot(x=df[col], y = df['salary'], data = df)
    plt.show()

#distribution plots 
def dis_plot(Rfunction, Bfunction, Rname, Bname, title):
    plt.figure(figsize=(14,6))
    ax1 = sb.distplot(Rfunction, hist = False, color = 'r', label = Rname)
    ax1 = sb.distplot(Bfunction, hist = False, color = 'b', label = Bname)
    plt.title(title)
    plt.show()
    plt.close()
    
#joining data set
def joinData(df1, df2, key=None, left_index=False, right_index=False):
    return pd.merge(left=df1, right=df2, how='inner', on=key, left_index=left_index, right_index=right_index)
    
    
#training models 
def trainModel (model, feat_df, tar_df, num_procs, mean_mse, cv_std):
    neg_mse = cross_val_score(model, feat_df, tar_df, cv=2, n_jobs=num_procs, scoring='neg_mean_sqaured_error')
    mean_mse[model]=-10*np.mean(neg_mse)
    cv_std[model]=np.std(neg_mse)


#print summary of models
def printSummary(model, mean_mse, cv_std):
    print('n\Model:\n', model)
    print('Average MSE:\n', mean_mse[model])
    print('Standard deviation during CV:\n', cv_std[model])

#saves results
def saveResults(model, mean_mse, predictions, feature_importances):
    with open('model.txt', 'w') as file:
        file.write(str(model))
    feature_importances.to_csv('feature_importances.csv') 
    np.savetxt('predictions.csv', predictions, delimiter=',')


# In[64]:


#Inputs
train_features = '/home/nimesh/salarypredictionportfolio/data/train_features.csv'
test_features = '/home/nimesh/salarypredictionportfolio/data/test_features.csv'
train_salaries = '/home/nimesh/salarypredictionportfolio/data/train_salaries.csv'

#Variables

column_cat = ['companyId', 'jobType', 'degree', 'major', 'industry']
column_num = ['yearsExperience', 'milesFromMetropolis']
target_col = 'salary'

#Load Data
print("Data is being Loaded")#quick indicator of load data 
train_features_df = loadFile(train_features)
test_features_df = loadFile(test_features)
train_salaries_df = loadFile(train_salaries)


# ## Part 2 - DISCOVER

# ### ---- 2 Load the data ----

# ### ---- 3 Clean the data ----

# In[65]:


#checking the first 10 rows of each data set to make sure data was loaded properly
train_features_df.head(10)


# In[66]:


test_features_df.head(10)


# In[67]:


train_salaries_df.head(10)


# In[68]:


#checking the basic details of the data sets 
train_features_df.info()
print()
test_features_df.info()
print()
train_salaries_df.info()


# In[69]:


#checking for duplicate data
print('Number of train feature duplicates: ' + str(train_features_df.duplicated().sum()))
print('Number of test feature duplicates: ' + str(test_features_df.duplicated().sum()))
print('Number of train salaries duplicates: ' + str(train_salaries_df.duplicated().sum()))


# In[70]:


#checking for NULL data points
train_features_df.isnull().sum()


# In[71]:


test_features_df.isnull().sum()


# In[72]:


train_salaries_df.isnull().sum()


# In[115]:


#look for duplicate data, invalid data (e.g. salaries <=0), or corrupt data and remove it


#joining the 2 train files on jobId
joined_df = joinData(train_features_df, train_salaries_df)


#checking for invalid data
print('Train Salaries that are negative: ' + str(sum(joined_df.salary < 0)))
print('Train Salaries that are zero: ' + str(sum(joined_df.salary == 0)))
print()

print('(Train)Years of Experience that are negative: ' + str(sum(joined_df.yearsExperience < 0)))
print('(Train)Years of Experience that are zero: ' + str(sum(joined_df.yearsExperience == 0)))
print('(Test)Years of Experience that are negative: ' + str(sum(test_features_df.yearsExperience < 0)))
print('(Test)Years of Experience that are zero: ' + str(sum(test_features_df.yearsExperience == 0)))
print() 

print('(Train)Miles from Metropolis that are negative: ' + str(sum(joined_df.milesFromMetropolis < 0)))
print('(Train)Miles from Metropolis that are zero: ' + str(sum(joined_df.milesFromMetropolis == 0)))
print('(Test)Miles from Metropolis that are negative: ' + str(sum(test_features_df.milesFromMetropolis < 0)))
print('(Test)Miles from Metropolis that are zero: ' + str(sum(test_features_df.milesFromMetropolis == 0)))
print()


# ### ---- 4 Explore the data (EDA) ----

# In[ ]:





# In[75]:


#summarizing numerical values
train_features_df.describe(include=[np.number])


# In[14]:


#summarizing categorical values 
train_features_df.describe(include=['O'])


# In[ ]:


#join training data
joined_df = joinData(train_features_df, train_salaries_df, key='jobId')


# In[79]:


#Checking joined df to see data types and if the data was joined properly
joined_df.info()
joined_df.head()


# In[80]:


#visualizing target variable 
plt.figure(figsize = (14, 6))
plt.subplot(1,2,1)
#to check for distribution of data points and potential outliers
sb.boxplot(joined_df.salary)
plt.subplot(1,2,2)
sb.distplot(joined_df.salary, bins=20 , color = 'blue')
plt.show()


# From the boxplot, we can see there are some potential outliers to investigate. We can also see the salaries have a symmetrical distribution. 

# In[81]:


#IQR to identify potential outliers 
stat = joined_df.salary.describe()
print(stat)


# In[82]:


IQR = stat['75%'] - stat['25%']
upper=stat['75%'] + 1.5*IQR
lower=stat['25%'] - 1.5*IQR
print("The upper bound for the suspected outliers is " + str(upper))
print("The lower bound for the suspected outliers is " + str(lower))


# In[83]:


#Examining potential outliers below lower bound
joined_df[joined_df.salary < 8.5]


# In[84]:


#Examining potential outliers above upper bound 
joined_df.loc[joined_df.salary > 222.5, 'jobType'].value_counts()


# In[85]:


#observing junior positions with salaries above the upper bound
joined_df[(joined_df.salary > 222.5) & (joined_df.jobType == 'JUNIOR')]


# In[86]:


#Drop data points with salary = 0 because they are missing salary values
joined_df = joined_df[joined_df.salary > 8.5]


# In[87]:


#looking for correlation between each feature and the target variable
#look for correlation between features


# In[88]:


plot_feat(joined_df, 'companyId')


# It is clear that companyId is not correlated with the salary amount which makes sense because there are different roles in a company. 

# In[89]:


plot_feat(joined_df, 'jobType')


# In[90]:


plot_feat(joined_df, 'degree')


# In[91]:


plot_feat(joined_df, 'major')


# In[92]:


plot_feat(joined_df, 'industry')


# In[93]:


plot_feat(joined_df, 'yearsExperience')


# In[94]:


plot_feat(joined_df, 'milesFromMetropolis')


# In[116]:


#cleaning and shuffling data (for better cross validation accuracy)
clean_train = shuffle(cleanData(joined_df)).reset_index()

#encode categorical data and get final feature dfs
print("Data is being Encoded")#quick indicator of dating being encoded
clean_train = one_hot_encoding(clean_train, cat_vars=column_cat, num_vars=column_num)
test_features_df = one_hot_encoding(test_features_df, cat_vars=column_cat, num_vars=column_num)

#target df
#train_salaries_df = clean_train['salary']


# In[96]:


for column in joined_df.columns:
    if joined_df[column].dtype.name == 'category':
        encode(joined_df,column)


# In[118]:


#correlation between selected features and response
#jobId is unique, therefore it is discarded because it does not play a role
figure = plt.figure(figsize=(14,12))
features = ['companyId', 'jobType', 'degree', 'major', 'industry', 'yearsExperience', 'milesFromMetropolis']
sb.heatmap(joined_df[column_num + column_cat].corr(), cmap='Blues', annot=True)
plt.xticks(rotation=45)
plt.show()


# ### ---- 5 Establish a baseline ----

# In[5]:


#select a reasonable metric (MSE in this case)
#create an extremely simple model and measure its efficacy
#e.g. use "average salary" for each industry as your model and then measure MSE
#during 5-fold cross-validation


# In[123]:



#Model initialization
models = []
mse_mean = {}
cv_std = {}
res = {}

#limting running processes in parallel
par_processes = 2

#shared model parameters
verbose_lvl = 5 

#create models - hyperparameter tuning done for each model
lin = LinearRegression()
lin_std_pca = make_pipeline(StandardScaler(), PCA(), LinearRegression())
randFor = RandomForestRegressor(n_estimators=60, n_jobs=par_processes, max_depth=25, min_samples_split=60, maxfeatures=30, verbose=verbose_lvl)
grad = GradientBoostingRegressor(n_estimators=40, max_depth=5, loss='ls', verbose=verbose_lvl)
models.extend([lin, lin_std_pca, randFor, grad])

#cross-validation of models in parallel, using MSE as evaluation metric and then printing their summaries 
print("Cross-Validation")
for model in models:
    trainModel (model, feat_df, tar_df, num_procs, mean_mse, cv_std)
    printSummary(model, mean_mse, cv_std)

#goal is to choose model with the lowest MSE
model = min(mean_mse, key=mean_mse.get)
print('\nPredictions calculated using model with lowest MSE \n' + model)

#use entire dataset to train model
model.fit(feat_df, tar_df)

#prediction based on test data set
predictions = model.predict(test_df)

#finding feature importances for each model and storing them
if hasattr(model, 'feature_importances_'):
    importance = model.feature_importances_
else:
    #for lineaer regression model
    importance = [0]*len(feat_df.columns)
    
feature_importances = pd.DataFrame({'feature': feat_df.columns, 'importance':importance})
feaure_importance.sort_values(by='importance', ascending=False, inplace=True)

#set index to 'feature'
feature_importance.set_index('feature', inplace=True, drop=True)

#save results
saveResults(model, mean_mse[model], predictions, feature_importances)


# In[ ]:





# ### ---- 6 Hypothesize solution ----

# In[ ]:


#brainstorm 3 models that you think may improve results over the baseline model based
#on your 


# Brainstorm 3 models that you think may improve results over the baseline model based on your EDA and explain why they're reasonable solutions here.
# 
# Also write down any new features that you think you should try adding to the model based on your EDA, e.g. interaction variables, summary statistics for each group, etc

# ## Part 3 - DEVELOP

# You will cycle through creating features, tuning models, and training/validing models (steps 7-9) until you've reached your efficacy goal
# 
# #### Your metric will be MSE and your goal is:
#  - <360 for entry-level data science roles
#  - <320 for senior data science roles

# ### ---- 7 Engineer features  ----

# In[ ]:


#make sure that data is ready for modeling
#create any new features needed to potentially enhance model


# ### ---- 8 Create models ----

# In[15]:


#create and tune the models that you brainstormed during part 2


# ### ---- 9 Test models ----

# In[1]:


#do 5-fold cross validation on models and measure MSE


# ### ---- 10 Select best model  ----

# In[ ]:


#select the model with the lowest error as your "prodcuction" model


# ## Part 4 - DEPLOY

# ### ---- 11 Automate pipeline ----

# In[ ]:


#write script that trains model on entire training set, saves model to disk,
#and scores the "test" dataset


# ### ---- 12 Deploy solution ----

# In[16]:


#save your prediction to a csv file or optionally save them as a table in a SQL database
#additionally, you want to save a visualization and summary of your prediction and feature importances
#these visualizations and summaries will be extremely useful to business stakeholders


# ### ---- 13 Measure efficacy ----

# We'll skip this step since we don't have the outcomes for the test data

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




