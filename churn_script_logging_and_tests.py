import os
import logging

from churn_library import *
from constants import DATA_FILE_PATH, LOG_FILE_PATH
#import churn_library as cls

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import():
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = import_data(FILE_PATH)
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(perform_eda):
	'''
	test perform eda function
	'''


def test_encoder_helper():
    '''
    test encoder helper
    '''
    df = import_data(FILE_PATH)

def test_one_hot_encoder():
    '''
    test encoder helper
    '''
    df = import_data(FILE_PATH)


def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''


def test_train_models(train_models):
	'''
	test train_models
	'''


if __name__ == "__main__":
    #pass
    test_import()
    #test_eda(perform_eda):
    test_encoder_helper()
    test_one_hot_encoder()
    #test_perform_feature_engineering(perform_feature_engineering):
    #test_train_models(train_models):

