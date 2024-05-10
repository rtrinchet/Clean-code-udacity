# pylint: disable=C0103
# disabling code to avoid naming like X or X_train

# library doc string
"""The churn_library.py is a library of functions to find customers
who are likely to churn
"""
import logging
logging.basicConfig(filename="logs/churn_log.log",
                    filemode="w",
                    level=logging.INFO,
                    format="%(name)s → %(levelname)s: %(message)s")
# import libraries

import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'

DATA_PATH = r"./data/bank_data.csv"
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


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    # logging.info('Importing data')
    dataf = pd.read_csv(pth)
    dataf['Churn'] = dataf['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    # logging.info('Done')
    return dataf


def perform_eda(df):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """
    # logging.info('Starting EDA')

    plt.figure(figsize=(20, 10))

    df['Churn'].hist()
    plt.savefig('images/churn.png')

    df['Customer_Age'].hist()
    plt.savefig('images/c_age.png')

    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('images/marital.png')

    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('images/ttrans.png')

    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('images/heatmap.png')
    # logging.info('Done')


def encoder_helper(df, category_lst):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15
    from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features

    output:
            df: pandas dataframe with new columns for
    """
    # logging.info('Encoding')
    print(df.shape)
    for cat_col in category_lst:
        temp_lst = []
        category_groups = df.groupby(cat_col).mean()['Churn']
        for val in df[cat_col]:
            temp_lst.append(category_groups.loc[val])

        df[f'{cat_col}_Churn'] = temp_lst
    # logging.info('Done')
    print(df.shape)

    return df


def perform_feature_engineering(df):
    """
    input:
              df: pandas dataframe

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    # logging.info('Doing F- Eng')

    df = encoder_helper(
        df,
        category_lst=[
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'])

    keep_cols = [
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
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X = pd.DataFrame()
    X[keep_cols] = df[keep_cols]
    y = df['Churn']
    # logging.info('Done')

    return train_test_split(X, y, test_size=0.3, random_state=42)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    """
    produces classification report for training and testing results and
    stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')
    # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')
    # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('images/classrep1.png')

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')
    # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')
    # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('images/classrep2.png')


def feature_importance_plot(model, X, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X.shape[1]), names, rotation=90)

    plt.savefig(output_pth)


def train_models(X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    # logging.info('TRAINING...')

    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    param_grid = {
        'n_estimators': [200, 500],
        'max_depth': [4],
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=2)

    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # scores
    # logging.info('random forest results')
    # logging.info('test results')
    # logging.info(classification_report(y_test, y_test_preds_rf))
    #
    # logging.info('train results')
    # logging.info(classification_report(y_train, y_train_preds_rf))
    #
    # logging.info('logistic regression results')
    # logging.info('test results')
    # logging.info(classification_report(y_test, y_test_preds_lr))
    # logging.info('train results')
    # logging.info(classification_report(y_train, y_train_preds_lr))

    lrc_plot = plot_roc_curve(lrc, X_test, y_test)

    # plots
    fig = plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)

    lrc_plot.plot(ax=ax, alpha=0.8)
    fig.savefig('images/train/roc_curve_1.png')

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    lrc_plot = plot_roc_curve(lr_model, X_test, y_test)

    fig = plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(rfc_model, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    fig.savefig('images/train/roc_curve_best_model.png')

    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    plt.savefig('images/train/shap_plot.png')

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    feature_importance_plot(cv_rfc, X_train, output_pth='images/fi_plot.png')


if __name__ == "__main__":
    dataframe = import_data(DATA_PATH)
    perform_eda(dataframe)
    X_Train, X_Test, y_Train, y_Test = perform_feature_engineering(dataframe)
    train_models(X_Train, X_Test, y_Train, y_Test)
