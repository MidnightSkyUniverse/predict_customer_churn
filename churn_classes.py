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
from sklearn.preprocessing import OneHotEncoder
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Initiate seaborn theme
sns.set()


class TrainModel():
    '''

    This class has three methods: fit, predict and save_model
    Model classes for Logistic Regression and Random Forest Classifier inherits from TrainModel

    '''

    def __init__(self):
        self.model = None

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_train, X_test, best_estimator):
        '''
        X: data
        y: target

        '''
        if (best_estimator==True):
            y_train_preds = self.model.best_estimator_.predict(X_train)
            y_test_preds = self.model.best_estimator_.predict(X_test)
        else:
            y_train_preds = self.model.predict(X_train)
            y_test_preds = self.model.predict(X_test)

        return (y_train_preds, y_test_preds)

    def save_model(self, pth):
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
    '''

        Class creates LogisticRegresion model object

    '''

    def __init__(self, random_state=None, max_iter=100):
        super().__init__()
        self.model = LogisticRegression(
            random_state=random_state, max_iter=max_iter)
        self.name = 'Logistic Regression'


class MyRandomForestClassifier(TrainModel):
    '''

        Class creates Random Forest Classifier model object
        It has two methods to define best_estimator_ and feature_importances_

    '''

    def __init__(self, param_grid, random_state=42, cv=5):
        super().__init__()
        self.rfc = RandomForestClassifier(random_state=random_state)
        self.model = GridSearchCV(
            estimator=self.rfc,
            param_grid=param_grid,
            cv=cv)
        self.name = 'Random Forest Classifier'



class DataEncoding():
    '''
    '''
    def __init__(self):
        self.data = pd.DataFrame()
        self.X = pd.DataFrame()
        self.y = None

    def import_data(self, pth):
        '''
        returns dataframe for the csv found at pth
        input:
            pth: a path to the csv
        output:
            df: pandas dataframe
        '''

        self.data = pd.read_csv(pth)

    def clean_data(self, new_col='Churn'):
        '''
        add column Churn that encodes column Existing Customer with 0 and 1
            input:
                df: pandas dataframe
            output:
                df: pandas dataframe
        '''
        self.data[new_col] = self.data['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

        labels_to_drop = [
            'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
            'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',
        ]
        self.data.drop(labels=labels_to_drop, axis=1, inplace=True)

    def encoder_helper(self, data, category_lst, keep_cols, response):
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
            new_col = col + '_' + response
            data[new_col] = col_lst

        self.X[keep_cols] = data[keep_cols]
        self.y = data[response]

    def one_hot_encoder(self, data, category_lst, response):
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
        cols_encoded = pd.DataFrame(enc.fit_transform(data[category_lst]))
        cols_remaining = data.drop(category_lst, axis=1)

        self.X = pd.concat([cols_encoded, cols_remaining], axis=1)
        self.y = data[response]


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

    def engineering(self, X_data, y_data, test_size=0.3, random_state=42):
        '''

            Function splts data into train and test data

            input:
                X_data, y_data
                test_size: how much data we want in test sample
                random_state: how much andomized the data can be
            output:
                X_train, y_train, X_test, y_test
        '''
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_data, y_data, test_size=test_size, random_state=random_state)

    def function_placeholder(self):
        pass


class MyFigure(plt.Figure):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #super(MyFigure, self).__init__(*args, **kwargs)
        self.fig = plt
        self.fig.figure(figsize=kwargs['figsize'])

    def save_figure(self, pth):
        self.fig.savefig(pth, bbox_inches='tight')
        self.fig.clf()

    def plot_histogram(self, data, col, title, pth):

        data[col].hist()
        self.fig.title(title)
        self.save_figure(pth)

    def plot_chart(self, data, col, title, pth, chart_type='bar'):

        data[col].value_counts('normalize').plot(kind=chart_type)
        self.fig.title(title)
        self.save_figure(pth)

    def plot_distribution_chart(self, data, col, title, pth):

        sns.displot(data[col])
        self.fig.title(title)
        self.save_figure(pth)

    def plot_heatmap(
            self,
            data,
            title,
            pth,
            cmap='Dark2_r',
            annot=False,
            linewidths=2):
        sns.heatmap(data.corr(), annot=annot, cmap=cmap, linewidths=linewidths)
        self.fig.title(title)
        self.save_figure(pth)

    def feature_importances(self, model, X_data, pth):
        '''
        creates and stores the feature importances in pth
        input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

        output:
             None
        '''
        # Calculate feature importances
        importances = model.best_estimator_.feature_importances_ 
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [X_data.columns[i] for i in indices]

        # Create plot title
        self.fig.title("Feature Importance")
        self.fig.ylabel('Importance')

        # Add bars
        self.fig.bar(range(X_data.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        self.fig.xticks(range(X_data.shape[1]), names, rotation=90)

        # Save figure
        self.save_figure(pth)


    def classification_report(self, model_name, y_train,
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
        self.fig.text(0.01, 0.95, str(f'{model_name} Train'), {
            'fontsize': 10}, fontproperties='monospace')
        self.fig.text(0.01, 0.55, str(classification_report(y_train, y_train_preds)), {
            'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        self.fig.text(0.01, 0.45, str(f'{model_name} Test'), {
            'fontsize': 10}, fontproperties='monospace')
        self.fig.text(0.01, 0.01, str(classification_report(y_test, y_test_preds)), {
            'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        self.fig.axis('off')
        self.save_figure(pth)

    def roc_curve_plot(self, model_1, model_2, X_test, y_test, pth):
        '''
        creates and stores the feature importances in pth
        input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

        output:
             None
        '''
        ax = plt.gca()
        lrc_plot = plot_roc_curve(model_1, X_test, y_test, ax=ax, alpha=0.8)
        lrc_plot = plot_roc_curve(model_2, X_test, y_test, ax=ax, alpha=0.8)
        self.save_figure(pth)

    def explainer_plot(self, model, X_test, plot_type, pth):
        '''

        This chart uses shap to add explainer        

        '''
        explainer = shap.TreeExplainer(model.best_estimator_)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, plot_type=plot_type, show=False)
        self.fig.title("Explainer bar chart")
        self.save_figure(pth)


if __name__ == '__main__':
    print('Churn classes')
