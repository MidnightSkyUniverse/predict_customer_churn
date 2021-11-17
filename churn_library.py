'''
    Project: 'Customer Churn Project' for Udacity nanodegree program

    Date: Nov 2021

    Author: Ali Binkowska
'''
# import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Import pcfg values
import projectconfig as pcfg
# Initiate seaborn theme
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''

    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # Histogram - customer churn column
    plt.figure(figsize=(20,10)) 
    df['Churn'].hist(); 
    plt.title("Histogram - Customer Churn")
    plt.savefig(pcfg.churn_distribution)

    # Histogram - customer agge distribution
    plt.figure(figsize=(20,10)) 
    df['Customer_Age'].hist();
    plt.title("Histogram - Customer Age")
    plt.savefig(pcfg.customer_age_distribution)

    # Bar chart - Material Status category
    plt.figure(figsize=(20,10)) 
    df.Marital_Status.value_counts('normalize').plot(kind='bar');
    plt.title("Bar chart - Marital status")
    plt.savefig(pcfg.material_status_distribution)
   
    # Distribution chart - Total transations
    plt.figure(figsize=(20,10)) 
    sns.displot(df['Total_Trans_Ct']); 
    plt.title("Distribution chart - Total transactions")
    plt.savefig(pcfg.total_transation_distribution)

    # Heatmap - all features
    plt.figure(figsize=(20,10)) 
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.title("Heatmap of all the features")
    plt.savefig(pcfg.heatmap)



def add_churn_column(df):
    '''
    add column Churn that encodes column Existing Customer with 0 and 1
    input:
        df: pandas dataframe
    output:
        df: pandas dataframe
    '''
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return df


def encoder_helper(df, category_lst, response):
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
        col_groups = df.groupby(col).mean()['Churn']

        for val in df[col]:
            col_lst.append(col_groups.loc[val])
        new_col = col + '_Churn'
        df[new_col] = col_lst

    X[pcfg.keep_columns] = df[pcfg.keep_columns]
    y = df['Churn']

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


def fit_random_forest_classifier(X_train, y_train, random_state=None):
    '''
    fitting of Random Forest Classifier model
    input:
              X_train: X training data
              y_train: y training data
              random_state: random states instant, default = None
    output:
              fitted model object
    '''
    rfc = RandomForestClassifier(random_state=random_state)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    return cv_rfc.best_estimator_


def fit_logistic_regression(X_train, y_train):
    '''
    fits Logistic Regression model
    input:
              X_train: X training data
              y_train: y training data
    output:
              fitted model object
    '''
    lrc = LogisticRegression()
    lrc.fit(X_train, y_train)

    return lrc


def predict_model(model, X_train, X_test):
    '''
    predicting of given model
    input:
              model: model that is fitted
              X_train: X training data
              X_test: X testing data
    output:
              y_train_preds: prediction on X training data
              y_test_preds : prediction on X testing data
    '''
    y_train_preds = model.predict(X_train)
    y_test_preds = model.predict(X_test)

    return (y_train_preds, y_test_preds)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds,
                                y_test_preds,
                                model_name,
                                output_pth):
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
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 0.95, str(f'{model_name} Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.55, str(classification_report(y_train, y_train_preds)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.45, str(f'{model_name} Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.01, str(classification_report(y_test, y_test_preds)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(output_pth)


def feature_importance_plot(model, X_data, output_pth):
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
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save figure
    plt.savefig(output_pth)


def roc_curve_plot(model_rf, model_lr, X_test, y_test, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    lrc_plot = plot_roc_curve(model_lr, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(
        model_rf,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(output_pth)


def explainer_plot(model, X_test, plot_type="bar"):
    '''
    '''
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type=plot_type, show=False)
    plt.title("Explainer bar chart")
    plt.savefig(pcfg.explainer)

def save_model(model, output_pth):
    '''
    saves model to ./models as .pkl file
    input:
            model: model object
            output_pth: path to store the model
    output:
             None
    '''
    joblib.dump(model, output_pth)


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    model_rfc = fit_random_forest_classifier(X_train, y_train, random_state=42)
    y_train_preds, y_test_preds = predict_model(model_rfc, X_train, X_test)

    model_name = 'Random Forest'
    classification_report_image(y_train,
                                y_test,
                                y_train_preds,
                                y_test_preds,
                                model_name, pcfg.rfc_results)

    model_lr = fit_logistic_regression(X_train, y_train)
    y_train_preds, y_test_preds = predict_model(model_lr, X_train, X_test)

    model_name = 'Logistic Regression'
    classification_report_image(y_train,
                                y_test,
                                y_train_preds,
                                y_test_preds,
                                model_name, pcfg.logistic_results)

    roc_curve_plot(
        model_rfc,
        model_lr,
        X_test,
        y_test,
        pcfg.roc_curve_result)
    
    # Save explainer bar plot
    explainer_plot(model_rfc, X_test, plot_type="bar")

    save_model(model_lr, pcfg.lr_model)
    save_model(model_rfc, pcfg.rfc_model)


if __name__ == '__main__':
    df = import_data(pcfg.data_file_path)
    df = add_churn_column(df)
    perform_eda(df)
    X, y  = encoder_helper(df, pcfg.cat_columns, response=None)
    #print(X.head())
    
    X_train, X_test, y_train, y_test = perform_feature_engineering(X, y, response=None)
    train_models(X_train, X_test, y_train, y_test)

    # Save Features Importances
    model_rfc = joblib.load(pcfg.rfc_model)
    feature_importance_plot(model_rfc, X, pcfg.feature_importances)
    # print(X_train.head())
