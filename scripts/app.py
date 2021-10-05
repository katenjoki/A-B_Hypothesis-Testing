import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.metrics import accuracy_score, confusion_matrix
                
header = st.container()
dataset = st.container()
model_build = st.container()

def get_data(filename):
    data = pd.read_csv(filename)
    return data

st.markdown(
'''
<style>
.main{
background-color:#F5F5F5;
}
</style>
''',unsafe_allow_html = True)
with header:
    st.title("A/B Hypothesis Testing: Ad campaign performance")
    st.text("The main objective of this project is to test if the ads that an advertising company run resulted in a significant lift in brand awareness.")

with dataset:
    st.header("Smart Ad data")
    st.text("The advertising company has a service called Brand Impact Optimiser (BIO), a lightweight questionnaire, served with every campaign to determine the \nimpact of the creative, the ad they design, on various upper funnel metrics,including memorability and brand sentiment.\n \nThe data for this project is a “Yes” and “No” response of online users to the following  question:\nQ: Do you know the brand Lux? \n O Yes O No")
    
    smart_data = get_data('data/AdSmartABdata.csv')
    st.write(smart_data.head())
    
    st.subheader("Plots of the features can be seen below! You can choose any pair of features to compare their distribution against each other")
        
    def plot_count(df,col1,col2):
        plt.figure(figsize=(12,8))

        plt.subplot(1,2,1)
        sns.countplot(data=df, x=col1,palette='summer')
        plt.title(f'Distribution of {col1}')
        plt.xticks(rotation=70)

        plt.subplot(1,2,2)
        sns.countplot(data=df, x=col2,palette='summer_r')
        plt.title(f'Distribution of {col2}')
        plt.xticks(rotation=50)
        plt.show()
            
    feature = ['Experiment vs Platform_OS','Yes vs No','Date vs Hour','Browser vs Platform']    
    feat = st.selectbox('What features should we compare distributions?',feature)
    if feat == 'Experiment vs Platform_OS':
        st.pyplot(plot_count(smart_data,'experiment','platform_os'))
    elif feat == 'Yes vs No':
        st.pyplot(plot_count(smart_data,'yes','no'))
    elif feat == 'Date vs Hour':
        st.pyplot(plot_count(smart_data,'date','hour'))
    elif feat == 'Browser vs Platform':
        st.pyplot(plot_count(smart_data,'browser','platform_os'))

with model_build:
    def loss_function(y_test, y_preds):
        rmse = np.sqrt(mean_squared_error(y_test, y_preds))
        r_sq = r2_score(y_test, y_preds)
        mae = mean_absolute_error(y_test, y_preds)
        st.write('Prediction RMSE Score: {}'.format(rmse))
        st.write('Prediction R2_Squared: {}'.format(r_sq))
        st.write('Prediction MAE Score: {}'.format(mae))
    
    st.header("Building models for the features!")
    feature = ['Browser','Platform']    
    feat = st.selectbox('What features should we focus our modelling on?',feature)
    if feat == 'Browser':
        browser = get_data('data/browser.csv')
        X=browser.loc[:,browser.columns != 'yes']
        y=browser['yes']
        X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7,random_state=42)
        models = ['Logistic Regression','Decision Tree Classifier','XGBoost']
        model = st.selectbox('What model should we build?',models)
        if model == 'Logistic Regression':
            logreg = LogisticRegression()
            logreg.fit(X_train,y_train)
            y_pred = logreg.predict(X_test)
            loss_function(y_test, y_pred)
        elif model == 'Decision Tree Classifier':
            dtree=DecisionTreeClassifier()
            dtree.fit(X_train,y_train)
            y_pred = dtree.predict(X_test)
            loss_function(y_test, y_pred)

        else:
            model=XGBClassifier()
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            loss_function(y_test, y_pred)
            
    else:
        platform = get_data('data/platform.csv')
        X=platform.loc[:,platform.columns != 'yes']
        y=platform['yes']
        X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7,random_state=42)
        models = ['Logistic Regression','Decision Tree Classifier','XGBoost']
        model = st.selectbox('What model should we build?',models)
        if model == 'Logistic Regression':
            logreg = LogisticRegression()
            logreg.fit(X_train,y_train)
            y_pred = logreg.predict(X_test)
            loss_function(y_test, y_pred)
        elif model == 'Decision Tree Classifier':
            dtree=DecisionTreeClassifier()
            dtree.fit(X_train,y_train)
            y_pred = dtree.predict(X_test)
            loss_function(y_test, y_pred)   
        else:
            model=XGBClassifier()
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            loss_function(y_test, y_pred)
            
