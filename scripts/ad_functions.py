#Loading useful packages
import re
import datetime
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import SelectFromModel

from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report

browser = pd.read_csv('browser.csv',encoding='unicode_escape')
platform = pd.read_csv('platform.csv',encoding='unicode_escape')

def scale_dataset(dataframe:pd.DataFrame,target_col='response'):
    cols = dataframe.columns.tolist()
    cols.remove(target_col)
    
    X = dataframe[cols].values
    y = dataframe[target_col].values
    #standardise data to have a uniform scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    #split the dataset
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
    #only oversample the train set
    #Oversampling generates new samples in the classes which are underepresented, so that they now match
    ros =RandomOverSampler()
    X_train,y_train = ros.fit_resample(X_train,y_train)
    #new_data = np.hstack((X,np.reshape(y,(-1,1))))
    #because we'll be using kfold validation, split the set into train and test
    return X_train,X_test,y_train,y_test

def loss_function(y_test, y_preds):
    #This is a classification problem. So metrics such as accuracy, precision, recall, f1 score are more appropriate
    st.write('--------------------------------------------------------------------')
    st.subheader('Classification Report')
    report = classification_report(y_test, y_preds)

    lines = report.split('\n')
    report_data = []
    for line in lines[2:-3]:  # Exclude headers and footer
        row = re.split(r'\s+', line.strip())
        report_data.append(row)
  
    report_df = pd.DataFrame(report_data, columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
    report_df.drop([2,3],axis=0,inplace=True)
    st.table(report_df)
    st.write('Accuracy score:', accuracy_score(y_test,y_preds))
    #st.table(classification_report(y_test,y_preds))
    st.write('--------------------------------------------------------------------')

def get_feature_importance(coefficients,is_browser=True):
    # Get feature names
    if is_browser:
        feature_names = browser.columns[browser.columns!='response']
    else:
        feature_names = platform.columns[platform.columns!='response']
    # Get feature coefficients (importance)
    feature_importance = pd.DataFrame({'feature':feature_names,'importance':coefficients})
    feature_importance.sort_values(by='importance',ascending=False,inplace=True)

    feature_importance = pd.DataFrame({'feature':feature_names,'importance':coefficients})
    return feature_importance.reindex(feature_importance['importance'].abs().sort_values(ascending=False).index)
    

def log_model(X_train,X_test,y_train,y_test,num_of_features=10,is_browser=True):
    kfold=KFold(n_splits=5,random_state=42,shuffle=True)
    model=LogisticRegression()
    #hyper-parameter tuning
    scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=kfold, n_jobs=-1)
    #print('Validation Accuracy: %.3f (%.3f)' %(np.mean(scores), np.std(scores)))
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    loss_function(y_test, y_pred)

    ##### FEATURE IMPORTANCE#####
    coefficients = model.coef_[0]
    feature_importance = get_feature_importance(coefficients,is_browser)
    #plot feature importance
    fig = px.bar(feature_importance[:num_of_features],x='feature',y='importance',title=f'Top {num_of_features} Features by Importance: Logistic Regression')
    return fig

def decision_trees(X_train,X_test,y_train,y_test,num_of_features=10,is_browser=True):
    kfold=KFold(n_splits=5,random_state=42,shuffle=True)
    dtree=DecisionTreeClassifier()
    #hyper-parameter tuning
    scores = cross_val_score(dtree, X_train, y_train, scoring='accuracy', cv=kfold, n_jobs=-1)
    #print('Validation Accuracy: %.3f (%.3f)' %(np.mean(scores), np.std(scores)))
    dtree.fit(X_train,y_train)
    y_pred = dtree.predict(X_test)
    loss_function(y_test, y_pred)
    
    importance = dtree.feature_importances_
    feature_importance = get_feature_importance(coefficients=importance,is_browser=is_browser)
    #plot feature importance
    fig = px.bar(feature_importance[:num_of_features],x='feature',y='importance',title=f'Top {num_of_features} Features by Importance: Decision Trees')
    return fig

def xgboost_model(X_train,X_test,y_train,y_test):
    kfold=KFold(n_splits=5,random_state=42,shuffle=True)
    model=XGBClassifier()
    scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=kfold, n_jobs=-1)
    #print('Validation Accuracy: %.3f (%.3f)' %(np.mean(scores), np.std(scores)))
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    loss_function(y_test, y_pred)
    
    # Fit model using each importance as a threshold
    thresholds = np.sort(model.feature_importances_)
    for thresh in thresholds:
        # select features using threshold
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train)
        # train model
        selection_model = XGBClassifier(verbosity=0)
        selection_model.fit(select_X_train, y_train)
        # eval model
        select_X_test = selection.transform(X_test)
        y_pred = selection_model.predict(select_X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        #print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
        
    return plot_importance(model,color='#B0C485')