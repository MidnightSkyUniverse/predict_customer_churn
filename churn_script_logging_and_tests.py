import os
import logging

# Project config and classes
import projectconfig as cfg
import churn_library as cls


logging.basicConfig(
    filename=cfg.log_file_path,
    level=logging.INFO,
    filemode='w',
    force=True,
    format='%(name)s - %(levelname)s - %(message)s')


def test_building_model():
    '''
    Test of class DataEncoding()
    '''
    try:
        data = cls.DataEncoding()
        isinstance(data, cls.DataEncoding)
        logging.info("Object DataEncoding() was created: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Object DataEncoding() not created")
        raise err

    try:
        data.import_data(cfg.data_file_path)
        logging.info("Testing import_data: SUCCESS")
    except AssertionError as err:
        logging.info("Testing import_data - file not found: FAILED")
        raise err

    try:
        assert data.data.shape[0] > 0
        assert data.data.shape[1] > 0
        logging.info("Testing shape of the imported data: SUCCESS")
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    
    try:
        data.clean_data('Churn')
        logging.info("Testing clean_data method: SUCCESS")
    except AssertionError as err:
        logging.error("Testing clean_data: issue during adding a new column and dropping two others")
        raise err
    
    try:
        data.encoder_helper(cfg.cat_columns, cfg.keep_columns, 'Churn')
        logging.info("Testing encoder_helper method: SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoder_helper: issue with setting X and y")
        raise err

    try:
        featured = cls.FeatureEngineering()
        isinstance(featured, cls.FeatureEngineering)
        logging.info("Object FeatureEngineering() was created: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Object FeatureEngineering() not created")
        raise err

    try:
        featured.engineering(data.X, data.y)
        logging.info("Testing engineering method: SUCCESS")
    except AssertionError as err:
        logging.error("Testing clean_data: issue during adding a new column and dropping two others")
        raise err

    try:
        model_lr = cls.MyLogisticRegression()
        isinstance(model_lr, cls.MyLogisticRegression)
        logging.info("Object MyLogisticRegression() was created: SUCCESS")
    except AssertionError as err:
        logging.error("Object MyLogisticRegression() was not created:")
        raise err

    try:
        featured.engineering(data.X, data.y)
        logging.info("Testing engineering method: SUCCESS")
    except AssertionError as err:
        logging.error("Testing clean_data: issue during adding a new column and dropping two others")
        raise err


        
    try:
        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }

        model_rfc = cls.MyRandomForestClassifier(param_grid)
        isinstance(model_rfc, cls.MyRandomForestClassifier)
        logging.info("Object MyRandomForestClassifier() was created: SUCCESS")
    except AssertionError as err:
        logging.error("Object MyRandomForestClassifier() was not created:")
        raise err


if __name__ == "__main__":
    pass
    # test_eda(perform_eda):
    # test_perform_feature_engineering(perform_feature_engineering):
    # test_train_models(train_models):

