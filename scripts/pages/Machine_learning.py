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

header = st.container()
dataset = st.container()
model_build = st.container()

browser = pd.read_csv('../data/browser.csv',encoding='unicode_escape')
platform = pd.read_csv('../data/platform.csv',encoding='unicode_escape')

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

def xgboost_model(X_train,X_test,y_train,y_test,num_of_features=10,is_browser=True):
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
    coefficients = model.feature_importances_
    feature_importance = get_feature_importance(coefficients=coefficients,is_browser=is_browser)
    #plot feature importance
    fig = px.bar(feature_importance[:num_of_features],x='feature',y='importance',title=f'Top {num_of_features} Features by Importance: XGBoost')
    return fig


with model_build:
    st.title("Machine Learning")
    st.text("With the features that we have, are we able to build a classification model that can\naccurately predict whether a user responded yes or no to the question:\nDo you know the brand Lux?")
    feature = ['Browser','Platform']    
    feat = st.selectbox('What features should we focus our modelling on?',feature,)
    if feat == 'Browser':
        is_browser=True
        data = pd.read_csv('browser.csv')
    else:
        data = pd.read_csv('platform.csv')
        is_browser = False
    X=data.loc[:,data.columns != 'response']
    y=data['response']
    X_train, X_test, y_train, y_test = scale_dataset(data)
    models = ['Logistic Regression','Decision Tree Classifier','XGBoost']
    model = st.selectbox('What model should we build?',models)
    number = st.number_input('Select number of features',min_value=1,max_value=X.shape[1],value=10)
    st.write('The current number of features is: ', number)
    if model == 'Logistic Regression':
        st.plotly_chart(log_model(X_train, X_test, y_train, y_test,num_of_features=number))
        st.markdown("- This model has a relatively poor performance, having an f1 score 0.52 for category 0.")
        st.markdown("- For category 0, a precision value of 0.52 means that of all the positive predictions that the model made, it only got 52% right.")
        st.markdown("- A recall of 0.53 means that of all the actual positives, the model only captured 53% ")
    elif model == 'Decision Tree Classifier':
        st.plotly_chart(decision_trees(X_train, X_test, y_train, y_test,num_of_features=number))
        st.markdown("- This model has a relatively poor performance, having an f1 score 0.56 for category 0.")
        st.markdown("- For category 0, a precision of 0.55 means that of all the positive predictions that the model made, it only got 55% right.")
        st.markdown("- A recall of 0.57 means that of all the actual positives, the model only captured 57% ")
    else:
        st.plotly_chart(xgboost_model(X_train, X_test, y_train, y_test,num_of_features=number))
        st.markdown("- This model has a relatively poor performance, having an f1 score 0.51 for category 0.")
        st.markdown("- For category 0, a precision of 0.52 means that of all the positive predictions that the model made, it only got 55% right.")
        st.markdown("- A recall of 0.51 means that of all the actual positives, the model only captured 57% ")
            
    st.subheader('Conclusion')
    st.markdown('- The data provided has few features that may not capture the complexity and variability present in the underlying data.')
    st.markdown('- Since the features may not have enough information to distinguish between different classes or predict the responses accurately, the models may struggle to generalize.')
