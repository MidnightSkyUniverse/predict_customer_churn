'''
    Tests of classes defined in churn_library.py for customer churn analyses

    to execute the tests use 'pyliny file_name.py'

'''
import logging

# Project config and classes
import projectconfig as cfg
import churn_library as cls


logging.basicConfig(
    filename=cfg.log_file_path,
    level=logging.INFO,
    filemode='w',
    force=True,
    #format='%(asctime)s: %(levelname)s - %(message)s')
    format='%(asctime)-15s  %(message)s')


def test_building_model():
    '''
    Test of class DataEncoding()
    '''
    # Test of data extraction and cleaning
    try:
        data = cls.DataEncoding()
        isinstance(data, cls.DataEncoding)
        logging.info("Object DataEncoding() was created: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Object DataEncoding() not created")
        raise err

    try:
        data.import_data(cfg.data_file_path)
        logging.info("Method import_data: SUCCESS")
    except AssertionError as err:
        logging.info("Method import_data - file not found: FAILED")
        raise err

    try:
        assert data.data.shape[0] > 0
        assert data.data.shape[1] > 0
        logging.info("Testing shape of the imported data: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    try:
        data.clean_data('Churn')
        logging.info("Method clean_data method: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Method clean_data: issue during adding a new column and dropping two others")
        raise err

    # Data prepping process
    try:
        data.encoder_helper(cfg.cat_columns, cfg.keep_columns, 'Churn')
        logging.info("Method encoder_helper method: SUCCESS")
    except AssertionError as err:
        logging.error("Method encoder_helper: issue with setting X and y")
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
        logging.info("Method engineering method: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Method clean_data: issue during adding a new column and dropping two others")
        raise err

    # Test of Logistic Regression model creation
    try:
        model_lr = cls.MyLogisticRegression()
        isinstance(model_lr, cls.MyLogisticRegression)
        logging.info("Object MyLogisticRegression() was created: SUCCESS")
    except AssertionError as err:
        logging.error("Object MyLogisticRegression() was not created:")
        raise err

    # Test of Logistic Regression model fitting
    try:
        model_lr.fit(featured.X_train, featured.y_train)
        logging.info("Method Logistic Regression fit method: SUCCESS")
    except AssertionError as err:
        logging.error("Method Logistic Regression fit: the process has failed")
        raise err

    # Test of Logistic Regression model predicting
    try:
        y_train_preds_lr, y_test_preds_lr = model_lr.predict(
            featured.X_train, featured.X_test, is_best_estimator=False)
        logging.info("Method Logistic Regression predict method: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Method Logistic Regression predict: the process has failed")
        raise err

     # Test of Logistic Regression model saving
    try:
        model_lr.save_model(cfg.lr_model)
        logging.info("Method Logistic Regression save method: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Method Logistic Regression save: the process has failed")
        raise err

    # Test of Random Forest Classifier model creation
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

    # Test of Random Forest Classifier model fitting
    try:
        model_rfc.fit(featured.X_train, featured.y_train)
        logging.info("Method Random Forest Classifier fit method: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Method Random Forest Classifier fit: the process has failed")
        raise err

    # Test of Random Forest Classifier model predicting
    try:
        y_train_preds_rfc, y_test_preds_rfc = model_rfc.predict(
            featured.X_train, featured.X_test, is_best_estimator=True)
        logging.info("Method Random Forest Classifier predict method: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Method Random Forest Classifier predict: the process has failed")
        raise err

    # Model saving
    try:
        model_rfc.save_model(cfg.rfc_model)
        logging.info("Method Random Forest Classifier save method: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Method Random Forest Classifier save: the process has failed")
        raise err

    try:
        model_lr.save_model(cfg.lr_model)
        logging.info("Method Logistic Regression save method: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Method Logistic Regression save: the process has failed")
        raise err

    # Model loading
    try:
        model_lr = cls.MyLogisticRegression()
        model_lr.load_model(cfg.lr_model)
        logging.info("Method Logistic Regression save method: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Method Logistic Regression save: the process has failed")
        raise err

    try:
        param_grid = {}
        model_rfc = cls.MyRandomForestClassifier(param_grid)
        model_rfc.load_model(cfg.rfc_model)
        logging.info("Method Random Forest Classifier save method: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Method Random Forest Classifier save: the process has failed")
        raise err

    # Tests for reports and charts
    try:
        eda = cls.MyFigure(figsize=(15, 8))
        eda.plot_histogram(
            data.data,
            'Churn',
            'Churn histogram',
            cfg.churn_distribution)
        eda.clf()
        logging.info("Figure plot_histogram: SUCCESS")
    except AssertionError as err:
        logging.error("Figure plot_histogram: the process has failed")
        raise err

    try:
        eda = cls.MyFigure(figsize=(15, 8))
        eda.plot_chart(
            data.data,
            'Marital_Status',
            "Bar chart - Marital status",
            cfg.marital_status_distribution,
            'bar')
        eda.clf()
        logging.info("Figure plot_chart: SUCCESS")
    except AssertionError as err:
        logging.error("Figure plot_chart: the process has failed")
        raise err

    try:
        eda = cls.MyFigure(figsize=(15, 8))
        eda.plot_distribution_chart(
            data.data,
            'Total_Trans_Ct',
            "Distribution chart - Total transactions",
            cfg.total_transation_distribution)
        eda.clf()
        logging.info("Figure plot_distribution_chart: SUCCESS")
    except AssertionError as err:
        logging.error("Figure plot_distribution_chart: the process has failed")
        raise err

    try:
        eda = cls.MyFigure(figsize=(15, 8))
        eda.plot_heatmap(data.data, 'Heatmap', cfg.heatmap)
        eda.clf()
        logging.info("Figure plot_heatmap: SUCCESS")
    except AssertionError as err:
        logging.error("Figure plot_heatmap: the process has failed")
        raise err

    try:
        fig = cls.MyFigure(figsize=(15, 8))
        fig.classification_report(
            model_lr.name,
            featured.y_train,
            featured.y_test,
            y_train_preds_lr,
            y_test_preds_lr,
            cfg.logistic_results)
        eda.clf()
        logging.info("Figure classification_report: SUCCESS")
    except AssertionError as err:
        logging.error("Figure classification_report: the process has failed")
        raise err

    try:
        eda = cls.MyFigure(figsize=(15, 8))
        eda.feature_importances(
            model_rfc.model,
            data.X,
            cfg.feature_importances)
        eda.clf()
        logging.info("Figure feature_importances: SUCCESS")
    except AssertionError as err:
        logging.error("Figure feature_importances: the process has failed")
        raise err

    try:
        eda = cls.MyFigure(figsize=(15, 8))
        eda.roc_curve_plot(
            model_lr.model,
            model_rfc.model.best_estimator_,
            featured.X_test,
            featured.y_test,
            cfg.roc_curve_result)
        eda.clf()
        logging.info("Figure roc_curve_plot: SUCCESS")
    except AssertionError as err:
        logging.error("Figure roc_curve_plot: the process has failed")
        raise err

    try:
        eda = cls.MyFigure(figsize=(15, 8))
        eda.explainer_plot(
            model_rfc.model,
            featured.X_test,
            "bar",
            cfg.explainer)
        eda.clf()
        logging.info("Figure explainer_plot: SUCCESS")
    except AssertionError as err:
        logging.error("Figure explainer_plot: the process has failed")
        raise err


if __name__ == "__main__":
    pass
    # test_eda(perform_eda):
    # test_perform_feature_engineering(perform_feature_engineering):
    # test_train_models(train_models):
