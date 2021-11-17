"""
    File: constants.py

    Constant values for churn_library module
"""
DATA_FILE_PATH = './data/BankChurners.csv'
LOG_FILE_PATH = './logs/churn_library.log'
#RESULTS_FILE_PATH ='./images/results'
LOGISTIC_RESULTS = './images/results/logistic_results.png'
RF_RESULTS = './images/results/rf_results.png'
ROC_CURVE_RESULT = './images/results/roc_curve_result.png'
FEATURE_IMPORTANCES = './images/results/feature_importances.png'
RFC_MODEL = './models/rfc_model.pkl'
LR_MODEL = './models/logistic_model.pkl'

CAT_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'                
]

QUANT_COLUMNS = [
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

KEEP_COLUMNS = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn'
]

