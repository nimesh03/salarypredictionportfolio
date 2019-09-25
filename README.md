# salarypredictionportfolio
Salary Prediction Portfolio Project (Python)

This project resolves a simple business problem related to HR. The goal is to use Machine Learning models to predict the salary of a worker based on various job features. 

Job Features include: 'companyId', 'jobType', 'degree', 'major', 'industry', 'yearsExperience', and 'milesFromMetropolis'
The target variable is: 'salary' 

This projects starts with Exploratory Data Analysis: checking for duplicates, zero values, examining potential outliers, correlation between the target variable and job features.

After exploring the data, three machine learning algorithms were selected with the goal to mimimize the Mean Sqaured Error (MSE): 1) Linear Regression 2) Random Forest 3) Gradient Boosting 

The models were created, hyperparameter tuning was done, and cross validtion was done in order to select the model with the lowest MSE
