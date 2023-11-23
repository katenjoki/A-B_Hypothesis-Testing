import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

import sys
import numpy as np
import pandas as pd

from scipy import stats
import plotly.express as px
from sklearn.metrics import classification_report
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

from ad_functions import *

header = st.container()
dataset = st.container()
model_build = st.container()


def preprocess_data(data):
    data.dropna(axis=0,inplace=True)
    #create a new df after dropping rows where both yes and no columns == 0
    data = data[~((data['yes']==0) & (data['no']==0))]
    #convert date columns to datetime
    data['date'] = pd.to_datetime(data['date'])
    #extract day column, may be there's a pattern
    data['day'] = pd.Series(data['date'].dt.day_name())
    #drop columns auction_id, no because that information is on column yes
    data.drop(['auction_id','no'],axis=1,inplace=True)
    #rename yes column to response where 1 means yes and 0 means no
    data.rename({'yes':'response'},axis=1,inplace=True)
    data.reset_index(drop=True, inplace=True)
    data.sample()
    return data

def plot_hist(dataframe,col1,feat):
    if feat == 'Date':
        dates = ['7/3/2020','7/4/2020','7/5/2020','7/6/2020','7/7/2020','7/8/2020','7/9/2020','7/10/2020']
        fig = px.histogram(smart_data,x=col1,barmode='group', color='response',
                    title=f'Number of responses by {feat}',category_orders={col1:dates})
    else:
        fig = px.histogram(dataframe,x=col1,barmode='group', color='response', title=f'Number of responses by {feat}')
    return fig
with header:
    st.title("A/B Hypothesis Testing: Ad campaign performance")
    st.text("The main objective of this project is to test if the ads that an advertising\ncompany run resulted in a significant lift in brand awareness.")

with dataset:
    st.header("Smart Ad data")
    st.text("The advertising company has a service called Brand Impact Optimiser (BIO), a\nlightweight questionnaire, served with every campaign to determine the impact of\nthe ad they design. The data for this project is a “Yes” and “No” response of\nonline users to the following question: \nDo you know the brand Lux? \n O Yes O No")
    df = pd.read_csv("../data/AdSmartABdata.csv")
    st.subheader("A snippet of the data")
    st.write(df.head())
    st.text("Response:\n1 - Yes\n0 - No")
    st.text("There are ids that have 0 for both yes and no - this means that a user saw the\nquestion and chose not to answer. These rows were dropped.")
    smart_data = preprocess_data(df)
    st.subheader('Visualization of the features')
    feature = ['Date','Platform OS','Browser']    
    feat = st.selectbox('Select a feature to visualize.',feature)
    
    if feat == 'Date':
        st.plotly_chart(plot_hist(smart_data,'date',feat))
        st.subheader("Observations")
        st.markdown("- 3rd July had the highest number of responses, it also happens to be a Friday and the first day of the experiment.")
        st.markdown("- 10th is also a Friday, but it had a lower response than\nWednesday and Thursday so we can't conclude with confidence that Fridays tend to have the highest number of responses.")
        st.markdown("- The marketing strategy was probably more aggressive\nat the beginning of the experiment.")
    elif feat == 'Platform OS':
        st.plotly_chart(plot_hist(smart_data,'platform_os',feat))
    elif feat == 'Browser':
        st.plotly_chart(plot_hist(smart_data,'browser',feat))    
    #st.plotly_chart(fig,use_container_width=True)

    st.subheader("Simple A/B test")
    st.text("Number of responses per group")
    st.table(pd.crosstab(smart_data['experiment'],smart_data['response']))
    st.text("Basic stats")
    response_rates = smart_data.groupby('experiment')['response'].agg([np.mean,np.std,stats.sem])
    response_rates.rename({'mean':'response_rate','std':'std_deviation','sem':'std_error'},axis=1,inplace=True)
    st.table(response_rates)
    st.markdown("- The exposed ad has a response rate of 46.8% while the control has 45.1%")
    st.markdown("- The exposed ad is doing slightly better than the control but is the difference statistically significant?")
    st.text("Z test")

    control = smart_data[smart_data.experiment == 'control'].response.values
    exposed = smart_data[smart_data.experiment != 'control'].response.values
    n_control = len(control)
    n_exposed = len(exposed)

    success = [control.sum(),exposed.sum()]
    nobs = [n_control,n_exposed]
    summary = pd.DataFrame()
    summary['experiment'] = ['control','exposed']
    summary['success'] = success
    summary['nobs'] = nobs
    st.table(summary)
    st.text("success:- this is the total sum of users who responded yes to the question")
    
    st.subheader("A/B test results")
    z_stat,pval = proportions_ztest(success,nobs)
    (lower_con, lower_exposed), (upper_con, upper_exposed) = proportion_confint(success,nobs,alpha=0.05)

    st.markdown(f'- z_statistic: {z_stat:.2f}')
    st.markdown(f'- p value: {pval:.3f}')
    st.markdown(f'- ci 95% for control group: [{lower_con:.3f},{upper_con:.3f}]')
    st.markdown(f'- ci 95% for exposed group: [{lower_exposed:.3f},{upper_exposed:.3f}]')
    st.subheader("Conclusion")
    st.text("The p value is 0.518, which is way above our significance level of 0.05.")
    st.text("We fail to reject the null hypothesis and conclude that the SmartAd design did not\nperform significantly different from the control and hence didn't lead to a\nsignificant increase in brand awareness.")                    

with model_build:    
    st.header("Machine Learning")
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
    number = st.number_input('Select number of features',min_value=1,max_value=len(X),value=10)
    st.write('The current number of features is: ', number)
    if model == 'Logistic Regression':
        st.plotly_chart(log_model(X_train, X_test, y_train, y_test,num_of_features=number))
        st.markdown("- This model has a relatively poor performance, having an f1 score 0.52.")
        st.markdown("- For category 0 , a precision of 0.52 means that of all the positive predictions that the model made, it only got 52% right.")
        st.markdown("- A recall of 0.53 means that of all the actual positives, the model only captured 53% ")
    elif model == 'Decision Tree Classifier':
        st.plotly_chart(decision_trees(X_train, X_test, y_train, y_test,num_of_features=number))
    else:
        st.plotly_chart(xgboost_model(X_train, X_test, y_train, y_test))
            
