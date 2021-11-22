'''
    This is a script that exacutes all project from command line.
    It will save the images and models which can be later viewed

    Date: Nov 2021

    Author: Ali Binkowska
'''
import projectconfig as cfg
import churn_library as cls


if __name__ == '__main__':
    encoded_data = cls.DataEncoding()
    encoded_data.import_data(cfg.data_file_path)
    encoded_data.clean_data('Churn')

    eda = [cls.MyFigure(figsize=(15, 8)) for i in range(5)]
    eda[0].plot_histogram(
        encoded_data.data,
        'Churn',
        'Churn histogram')
    eda[0]. save_figure(cfg.churn_distribution)

    eda[1].plot_histogram(
        encoded_data.data,
        'Customer_Age',
        "Histogram - Customer Age")
    eda[1].save_figure(cfg.customer_age_distribution)

    eda[2].plot_chart(
        encoded_data.data,
        'Marital_Status',
        "Bar chart - Marital status",
        'bar')
    eda[2].save_figure(cfg.marital_status_distribution)

    eda[3].plot_distribution_chart(
        encoded_data.data,
        'Total_Trans_Ct',
        "Distribution chart - Total transactions")
    eda[3].save_figure(cfg.total_transation_distribution)

    eda[4].plot_heatmap(encoded_data.data, 'Heatmap')
    eda[4].save_figure(cfg.heatmap)
    


    encoded_data.encoder_helper(cfg.cat_columns, cfg.keep_columns, 'Churn')
    featured_data = cls.FeatureEngineering()
    featured_data.engineering(encoded_data.X, encoded_data.y)

    model_lr = cls.MyLogisticRegression()
    model_lr.fit(featured_data.X_train, featured_data.y_train)
    y_train_preds, y_test_preds = model_lr.predict(
        featured_data.X_train, featured_data.X_test, is_best_estimator=False)
    model_lr.save_model(cfg.lr_model)

    fig = cls.MyFigure(figsize=(15, 8))
    fig.classification_report(
        model_lr.name,
        featured_data.y_train,
        featured_data.y_test,
        y_train_preds,
        y_test_preds)
    fig.save_figure(cfg.logistic_results)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # Declare model
    model_rfc = cls.MyRandomForestClassifier(param_grid)
    # Fit model
    model_rfc.fit(featured_data.X_train, featured_data.y_train)
    # Predict model
    y_train_preds, y_test_preds = model_rfc.predict(
        featured_data.X_train, featured_data.X_test, is_best_estimator=True)
    model_rfc.save_model(cfg.rfc_model)

    fig = cls.MyFigure(figsize=(15, 8))
    fig.classification_report(
        model_rfc.name,
        featured_data.y_train,
        featured_data.y_test,
        y_train_preds,
        y_test_preds)
    fig.save_figure(cfg.rfc_results)

    # Feature importances plot
    fig = cls.MyFigure(figsize=(15, 8))
    fig.feature_importances(
        model_rfc.model,
        encoded_data.X)
    fig.save_figure(cfg.feature_importances)

    fig = cls.MyFigure(figsize=(15, 8))
    fig.roc_curve_plot(
        model_lr.model,
        model_rfc.model.best_estimator_,
        featured_data.X_test,
        featured_data.y_test)
    fig.save_figure(cfg.roc_curve_result)

    fig = cls.MyFigure(figsize=(15, 8))
    fig.explainer_plot(
        model_rfc.model,
        featured_data.X_test,
        "bar")
    fig.save_figure(cfg.explainer)


