'''
    Classes for Churn project

    Date: Nov 2021

    Author: Ali Binkowska
'''
# import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
#import shap
#import joblib
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Import pcfg values
#import projectconfig as pcfg
# Initiate seaborn theme
sns.set()


class MyLogisticRegression():

    def __init__(self,random_state=None):
        self.random_state = random_state
        self.max_iter = 100
        self = LogisticRegression()

    def fit(self, X_data, y_target):
        '''
        Fit Logistic Regression model

        X: data
        y: target

        '''
        self.fit(X_data, y_target)

        return self

class MyRandomForestClassifier():

    def __init__(self,random_state, parameter_grid):
        self.random_state = random_state
        self.parameter_grid = parameter_grid 
        self = RandomForestClassifier(random_state=self.random_state)

    def fit():
        self =  GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)


class FeatureEngineerin():
    
    def __init__(self):
        pass

    def encoder_helper(data_frame, category_lst, keep_cols, response):
        '''
        helper function to turn each categorical column into a new column with
        propotion of churn for each category - associated with cell 15 from the notebook

        input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used
                    for naming variables or index y column]

        output:
            df: pandas dataframe with new columns for
        '''
        X = pd.DataFrame()
        for col in category_lst:
            col_lst = []
            col_groups = df.groupby(col).mean()[response]

            for val in df[col]:
                col_lst.append(col_groups.loc[val])
            new_col = col + response
            df[new_col] = col_lst

        X[keep_cols] = df[keep_cols]
        y = df[response]

        return (X,y)


    def one_hot_encoder(df, category_lst, response):
        '''
        helper function to turn each categorical column into a new column with one-hot encoder

        input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used
                    for naming variables or index y column]

        output:
            df: pandas dataframe with new columns for
        '''
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        cols_encoded = pd.DataFrame(enc.fit_transform(df[category_lst]))
        cols_remaining = df.drop(category_lst, axis=1)

        encoded_df = pd.concat([cols_encoded, cols_remaining], axis=1)

        return encoded_df


    def perform_feature_engineering(X,y, response):
        '''
        input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used
                        for naming variables or index y column]

        output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
        '''

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        return (X_train, X_test, y_train, y_test)


class MyFigure(plt.Figure):

    def __init__(self, figsize=(20, 10)):
        #super().__init__(figsize=figsize)
        self = plt.figure(figsize=figsize)

    def save_figure(self, plt, pth):
        plt.savefig(pth, bbox_inches='tight')
        plt.clf()


    def plot_histogram(self, df, col, title, pth):

        df[col].hist()
        plt.title(title)
        self.save_figure(plt, pth)

    def plot_chart(self, df, col, title, pth, chart_type='bar'):

        df[col].value_counts('normalize').plot(kind=chart_type)
        plt.title(title)
        self.save_figure(plt, pth)

    def plot_distribution_chart(self, df, col, title, pth):

        sns.displot(df[col])
        plt.title(title)
        self.save_figure(plt, pth)

    def plot_heatmap(
            self,
            df,
            title,
            pth,
            cmap='Dark2_r',
            annot=False,
            linewidths=2):
        sns.heatmap(df.corr(), annot=annot, cmap=cmap, linewidths=linewidths)
        plt.title(title)
        self.save_figure(plt, pth)


if __name__ == '__main__':
    pass
#    mlr = MyLogisticRegression()
