"""
    File: constants.py

    Constant values for churn_library module
"""
data_file_path = './data/BankChurners.csv'
log_file_path = './logs/churn_library.log'
#results_pth ='./images/results'
logistic_results = './images/results/logistic_results.png'
rfc_results = './images/results/rf_results.png'
roc_curve_result = './images/results/roc_curve_result.png'
feature_importances = './images/results/feature_importances.png'
rfc_model = './models/rfc_model.pkl'
lr_model = './models/logistic_model.pkl'
churn_distribution = './images/eda/churn_distribution.png'
customer_age_distribution = './images/eda/customer_age_distribution.png'
heatmap = './images/eda/churn_distribution.png'
material_status_distribution = './images/eda/material_status_distribution.png'
total_transation_distribution = './images/eda/total_transation_distribution.png'
explainer = './images/results/explainer.png'



cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'                
]

quant_columns = [
    'Customer_Age',
    'Dependent_count', 
    'Months_on_book',
    'Total_Relationship_Count', 
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 
    'Credit_Limit', 
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 
    'Total_Amt_Chng_Q4_Q1', 
    'Total_Trans_Amt',
    'Total_Trans_Ct', 
    'Total_Ct_Chng_Q4_Q1', 
    'Avg_Utilization_Ratio'
]

keep_columns = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn'
]

