# What Is This?

This is simple machine learning program for Udacity DevOps training.
The code is based on the code provided by Udacity.
The project is called 'Customer Churn', it uses kaggle dataset to predict 
customer churn using two models:
	- Random Forest Classifier
	- Logistic Regression


# How To Use This

## Requirements

There is requirements.txt uploaded to the repository
The program was installed on anaconda

## Prepare environment

	File projectconfig.py contains file paths to where the reports and charts will be stored.
	Folder to store models and figures must be create prior to execution of the project
	- ./images/eda
	- ./images/results
	- ./models
	- ./logs	
	- ./data 
	However if you decide to run the project from jupyter, the charts and reports are 
	printed to the screen and are not saved

### Dataset

	Dataset need to be downloaded from Kaggle:
	
		https://www.kaggle.com/sakshigoyal7/credit-card-customers/code

	and stored under ./data and the path updated in project config file

### Execution of the program

	Use churn_notebook.ipynb for sample usage example
	Please note that jupyter notebook does not save any output, while the churn_library.py
	executed from command line would require the folders created for images and models

# Testing

1. Run the command `pytest churn_script_logging_and_tests_solution.py`
2. Log file ./logs/churn_library.log is generated every time the test is launched
