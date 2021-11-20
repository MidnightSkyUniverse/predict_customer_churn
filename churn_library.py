'''
    Project: 'Customer Churn Project' for Udacity nanodegree program

    Date: Nov 2021

    Author: Ali Binkowska
'''
# import libraries
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import plot_roc_curve, classification_report
#import shap
#import joblib
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
# Import pcfg values
import projectconfig as pcfg
# Initiate seaborn theme
#sns.set()
import churn_classes as cls



if __name__ == '__main__':
    '''
        comman line tests of churn_classes
    '''
    ml_data = cls.DataEncoding()
    ml_data.import_data(pcfg.data_file_path)
    ml_data.clean_data('Churn')
    
    eda = [cls.MyFigure(figsize=(15,8)) for i in range(5)]
    eda[0].plot_histogram(ml_data.data, 'Churn', 'Churn histogram',pcfg.churn_distribution)
    eda[1].plot_histogram(ml_data.data, 'Customer_Age', "Histogram - Customer Age" ,pcfg.customer_age_distribution)
    eda[2].plot_chart(ml_data.data, 'Marital_Status', "Bar chart - Marital status", pcfg.marital_status_distribution, 'bar')
    eda[3].plot_distribution_chart(ml_data.data, 'Total_Trans_Ct', "Distribution chart - Total transactions", pcfg.total_transation_distribution)
    eda[4].plot_heatmap(ml_data.data, 'Heatmap', pcfg.heatmap )#annot=False, cmap='Dark2_r', linewidths = 2, pth=pcfg.heatmap)
        

    encoded_data = cls.DataEncoding()
    encoded_data.encoder_helper(ml_data.data, pcfg.cat_columns, pcfg.keep_columns, 'Churn')
    featured_data = cls.FeatureEngineering()
    featured_data.engineering(encoded_data.X,encoded_data.y)
        
    model_lr = cls.MyLogisticRegression()
    model_lr.fit(featured_data.X_train, featured_data.y_train)    
    y_train_preds, y_test_preds = model_lr.predict(featured_data.X_train, featured_data.X_test, best_estimator=False)
    model_lr.save_model(pcfg.lr_model)

    
    classification_figure_lr = cls.MyFigure(figsize=(15,8))
    classification_figure_lr.classification_report(model_lr.name, featured_data.y_train, featured_data.y_test, y_train_preds, y_test_preds,pcfg.logistic_results)   
    

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # Declare model
    model_rfc =  cls.MyRandomForestClassifier(param_grid)
    # Fit model
    model_rfc.fit(featured_data.X_train, featured_data.y_train)
    # Predict model
    y_train_preds, y_test_preds = model_rfc.predict(featured_data.X_train, featured_data.X_test, best_estimator=True)
    model_rfc.save_model(pcfg.rfc_model)
     
    classification_figure_rfc = cls.MyFigure(figsize=(15,8))
    classification_figure_rfc.classification_report(model_rfc.name, featured_data.y_train, featured_data.y_test, y_train_preds, y_test_preds,pcfg.rfc_results)   


    # Feature importances plot
    importances_plot = cls.MyFigure(figsize=(15,8))
    importances_plot.feature_importances(model_rfc.model, encoded_data.X, pcfg.feature_importances)
    
    roc = cls.MyFigure(figsize=(15,8))
    roc.roc_curve_plot(model_lr.model, model_rfc.model.best_estimator_, featured_data.X_test, featured_data.y_test, pcfg.roc_curve_result)

    expl = cls.MyFigure(figsize=(15,8))
    expl.explainer_plot(model_rfc.model, featured_data.X_test, "bar", pcfg.explainer)

    

