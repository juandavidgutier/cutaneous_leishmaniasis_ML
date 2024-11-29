#!/usr/bin/env python
# coding: utf-8

#importing libraries
import numpy as np
import shap
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, precision_recall_curve, f1_score, accuracy_score, precision_score, recall_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_line, ggtitle, labs, geom_point, geom_hline, theme_linedraw, theme, element_rect, theme_light, element_line, element_text
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

#%%#

#import dataset
df = pd.read_csv('D:/clases/UDES/articulo leishmaniasis/shap/dataset.csv')
df = df.drop(['DANE', 'Year', 'Month',
              'Cases', 'Population'], axis=1)
df = df.dropna()

X = df.loc[:, df.columns != 'excess']
y = df[['excess']]

names_X = list(X.columns.values.tolist())

X = pd.DataFrame(X)
y = pd.DataFrame(y)

#scale 0 to 1
Xscaler = MinMaxScaler(feature_range=(0, 1)) # scale so that all the X data will range from 0 to 1
Xscaler.fit(X)
X = Xscaler.transform(X)

#train_test_split 
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=0)

#%%#
#random forest

#training model 
rf=RandomForestClassifier(n_estimators=20000, max_depth=6, n_jobs=-2, random_state=123) 
rf.fit(x_train, y_train)

#now predicting model on test set 
y_pred=rf.predict(x_test) 

#precision recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
auc_precision_recall = auc(recall, precision)
print(auc_precision_recall)

# Create a DataFrame for plotting
pr_df = pd.DataFrame({'precision': precision, 'recall': recall})

# Plot the precision-recall curve using plotnine
precision_recall_plot = (
    ggplot(pr_df, aes(x='recall', y='precision')) +
    geom_line(color='red') +
    labs(title='', x='Recall', y='Precision') +
    theme(panel_background=element_rect(fill='lightblue'),
            axis_text_x=element_text(size=12), 
            axis_text_y=element_text(size=12),
            axis_title_x=element_text(size=15),
            axis_title_y=element_text(size=15),
            plot_title=element_text(size=18))
)

# Display the plot
print(precision_recall_plot)

#other metrics
#Converting probabilities into 1 or 0  
for i in range(0,31066): 
    if y_pred[i]>=.5:       # setting threshold to .5 
       y_pred[i]=1 
    else: 
       y_pred[i]=0  
    
print(confusion_matrix(y_pred, y_test))
# precision tp / (tp + fp)
precision = precision_score(y_test,y_pred)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test,y_pred)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test,y_pred)
print('F1 score: %f' % f1)  
print('auc_roc: %f' % roc_auc_score(y_pred, y_test))
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test,y_pred)
print('Accuracy: %f' % accuracy)


#%%#
#neural network

#training model 
nn=MLPClassifier(hidden_layer_sizes=(2000,1000,100), max_iter=1000, alpha=0.0001,
                 solver='sgd', verbose=10, tol=0.000000001, random_state=123) 
nn.fit(x_train, y_train)

#now predicting model on test set 
y_pred=nn.predict(x_test) 

#precision recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
auc_precision_recall = auc(recall, precision)
print(auc_precision_recall)

# Create a DataFrame for plotting
pr_df = pd.DataFrame({'precision': precision, 'recall': recall})

# Plot the precision-recall curve using plotnine
precision_recall_plot = (
    ggplot(pr_df, aes(x='recall', y='precision')) +
    geom_line(color='red') +
    labs(title='', x='Recall', y='Precision') +
    theme(panel_background=element_rect(fill='lightblue'),  
            axis_text_x=element_text(size=12), 
            axis_text_y=element_text(size=12),
            axis_title_x=element_text(size=15),
            axis_title_y=element_text(size=15),
            plot_title=element_text(size=18))
)

# Display the plot
print(precision_recall_plot)

#other metrics
#Converting probabilities into 1 or 0  
for i in range(0,31066): 
    if y_pred[i]>=.5:       # setting threshold to .5 
       y_pred[i]=1 
    else: 
       y_pred[i]=0  
    
print(confusion_matrix(y_pred, y_test))
# precision tp / (tp + fp)
precision = precision_score(y_test,y_pred)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test,y_pred)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test,y_pred)
print('F1 score: %f' % f1)  
print('auc_roc: %f' % roc_auc_score(y_pred, y_test))
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test,y_pred)
print('Accuracy: %f' % accuracy)


#%%#
#xgb
random.seed(123)

#The data is stored in a DMatrix object 
#label is used to define our outcome variable
dtrain=xgb.DMatrix(x_train,label=y_train)
dtest=xgb.DMatrix(x_test)

parameters={'max_depth':6, 'eta':0.1, 'objective':'binary:logistic', 'eval_metric':'aucpr', 'learning_rate':0.001}

#training model 
num_round=20000
xg=xgb.train(parameters ,dtrain, num_round) 

#now predicting model on test set 
y_pred=xg.predict(dtest) 

#precision recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
auc_precision_recall = auc(recall, precision)
print(auc_precision_recall)

# Create a DataFrame for plotting
pr_df = pd.DataFrame({'precision': precision, 'recall': recall})

# Plot the precision-recall curve using plotnine
precision_recall_plot = (
    ggplot(pr_df, aes(x='recall', y='precision')) +
    geom_line(color='red') +
    labs(title='', x='Recall', y='Precision') +
    theme(panel_background=element_rect(fill='lightblue'),  
            axis_text_x=element_text(size=12), 
            axis_text_y=element_text(size=12),
            axis_title_x=element_text(size=15),
            axis_title_y=element_text(size=15),
            plot_title=element_text(size=18))
)

# Display the plot
print(precision_recall_plot)

#other metrics
#Converting probabilities into 1 or 0  
for i in range(0,31066): 
    if y_pred[i]>=.5:       # setting threshold to .5 
       y_pred[i]=1 
    else: 
       y_pred[i]=0  
    
print(confusion_matrix(y_pred, y_test))
# precision tp / (tp + fp)
precision = precision_score(y_test,y_pred)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test,y_pred)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test,y_pred)
print('F1 score: %f' % f1)  
print('auc_roc: %f' % roc_auc_score(y_pred, y_test))
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test,y_pred)
print('Accuracy: %f' % accuracy)


#%%#
#shap values

#select subset for shap
sel = np.random.choice(x_train.shape[0], size=5000, replace=False)
x_sel = x_train[sel]

# explain the model's predictions using SHAP
explainer = shap.TreeExplainer(xg)
shap_values = explainer.shap_values(x_sel)

# summarize the effects of all the features
shap.summary_plot(shap_values, x_sel, names_X)

# create a dependence plot 
shap.dependence_plot("rank(0)", shap_values, x_sel, names_X, x_jitter=1, dot_size=2.5)
shap.dependence_plot("rank(1)", shap_values, x_sel, names_X, x_jitter=1, dot_size=2.5)
shap.dependence_plot("rank(2)", shap_values, x_sel, names_X, x_jitter=1, dot_size=2.5)
shap.dependence_plot("rank(3)", shap_values, x_sel, names_X, x_jitter=1, dot_size=2.5)
shap.dependence_plot("rank(4)", shap_values, x_sel, names_X, x_jitter=1, dot_size=2.5)
shap.dependence_plot("rank(5)", shap_values, x_sel, names_X, x_jitter=1, dot_size=2.5)
