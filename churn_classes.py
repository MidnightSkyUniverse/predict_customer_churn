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
import joblib
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Initiate seaborn theme
sns.set()


class TrainModel():
    '''

    Both models Logistic Regression and Random Forest Classifier inherits methods form this class

    '''
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_train, X_test):
        '''
        X: data
        y: target

        '''
        y_train_preds = self.model.predict(X_train)
        y_test_preds = self.model.predict(X_test)

        return (y_train_preds, y_test_preds)

    def save_model(self,pth):
        '''
             saves model to ./models as .pkl file
                input:
                    model: model object
                    output_pth: path to store the model
                output:
                    None
        '''
        joblib.dump(self.model, pth)



class MyLogisticRegression(TrainModel):

    def __init__(self,random_state=None,max_iter=100):
        self.model = LogisticRegression(random_state=random_state, max_iter=max_iter)
        self.name = 'Logistic Regression'


class MyRandomForestClassifier(TrainModel):

    def __init__(self,param_grid, random_state=42, cv=5):
        self.rfc = RandomForestClassifier(random_state=random_state)
        self.model =  GridSearchCV(estimator=self.rfc, param_grid=param_grid, cv=cv)
        self.name = 'Random Forest Classifier'
        

class DataEncoding():
    
    def __init__(self):
        self.X = pd.DataFrame()
        self.y = None

    def encoder_helper(self,data, category_lst, keep_cols, response):
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
        for col in category_lst:
            col_lst = []
            col_groups = data.groupby(col).mean()[response]

            for val in data[col]:
                col_lst.append(col_groups.loc[val])
            new_col = col +'_'+ response
            data[new_col] = col_lst

        self.X[keep_cols] = data[keep_cols]
        self.y = data[response]

    def one_hot_encoder(self, df, category_lst, response):
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

class FeatureEngineering():
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
    def __init__(self):
    #    '''
    #    '''
        self.X_train = None
        self.X_test = None  
        self.y_train = None 
        self.y_test = None

    def engineering(self,X,y, test_size=0.3, random_state=42):    
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state)




    

class MyFigure(plt.Figure):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

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

    def classification_report(self,model_name, y_train,
                                y_test,
                                y_train_preds,
                                y_test_preds,
                                pth):
        '''
        produces classification report for training and testing results and stores report as image
        in images folder
        input:
            y_train: training response values
            y_test:  test response values
            y_train_preds: training predictions from a model
            y_test_preds: test predictions from logistic regression
            model_name: name of a model
            image_name: name of image to store as .png file
        output:
             None
        '''
        #plt.rc('figure', figsize=(5, 5))
        plt.text(0.01, 0.95, str(f'{model_name} Train'), {
             'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.55, str(classification_report(y_train, y_train_preds)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.text(0.01, 0.45, str(f'{model_name} Test'), {
             'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.01, str(classification_report(y_test, y_test_preds)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.axis('off')
        self.save_figure(plt, pth)



if __name__ == '__main__':
    pass
#    mlr = MyLogisticRegression()
