# pylint: disable=C0103
# disabling code to avoid naming like X or X_train

"""
Module that tests the functions in churn_library
"""
import os
import logging
import glob

import pytest
import joblib
from sklearn.metrics import accuracy_score
from churn_library import (import_data, perform_eda,
                           encoder_helper, perform_feature_engineering,
                           train_models)


logging.debug('Starting test run')
FOLDER_PATH = "images"


@pytest.fixture
def import_data_fixture():
    """
    Fixture that performs data import
    """
    try:
        df = import_data("./data/bank_data.csv")
        logging.debug("Testing import_data: SUCCESS")
        print('Initially imported dataframe shape', df.shape)
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err
    return df


@pytest.fixture
def eda(import_data_fixture):
    """
    Fixture that performs EDA step
    """
    # delete png images:
    # checks images are generated
    delete_png_images(FOLDER_PATH)

    df = import_data_fixture
    perform_eda(df)

    # target is to check if images are generated in this function
    return df


@pytest.fixture
def clean_folder():
    """
    Cleans the folder
    """
    yield
    delete_png_images(FOLDER_PATH)


@pytest.fixture
def df_with_encoder(import_data_fixture):
    """
    Fixture that performs encoding on the input df
    """
    feature_list = ['Gender', 'Education_Level', 'Marital_Status',
                    'Income_Category', 'Card_Category']
    df = import_data_fixture
    df = encoder_helper(df, category_lst=feature_list)
    return df


@pytest.fixture
def feature_engineered_data(import_data_fixture):
    """
    Fixture that performs feature engineering
    """
    df = import_data_fixture
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    return X_train, X_test, y_train, y_test


def check_pkl_files(path):
    """
    Checks if there are pkl files in path
    """
    pkl_files = glob.glob(path + "/*.pkl")
    return len(pkl_files) > 0


def delete_png_images(FOLDER_PATH):
    """
    Deletes the png images from the specified folder.
    Used in order to test that EDA creates the images correctly
    """
    for filename in os.listdir(FOLDER_PATH):
        if filename.endswith(".png"):
            file_path = os.path.join(FOLDER_PATH, filename)
            os.remove(file_path)
            logging.debug(f"Deleted {filename}")


def test_import(import_data_fixture):
    """
    test data import - this example is completed for you to assist with
    the other test functions
    """
    df = import_data_fixture

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have "
            "rows and columns")
        raise err

    assert 'Churn' in df.columns, logging.error(
        'Churn column is not included in data')

    logging.info('test_eda was successful')


def test_eda(eda, clean_folder):
    """
    test perform eda function
    """
    # now check if the images are generated after running EDA step
    png_files = glob.glob(FOLDER_PATH + "/*.png")

    assert len(png_files) > 0, logging.error(
        "Folder does not contain PNG files!")
    logging.debug("Folder contains PNG files!")

    logging.info('test_eda was successful')


def test_encoder_helper(df_with_encoder):
    """
    test encoder helper
    """
    df_enc = df_with_encoder
    # using the logic for the creation of the encoded columns
    encoded_columns = [c for c in df_enc.columns if '_Churn' in c]
    assert len(encoded_columns) == 5, logging.error(
        'There is a problem with the column creation for encoder helper. '
        'A wrong number of columns was created\n'
    )
    logging.info('test_encoder_helper was successful')


def test_perform_feature_engineering(feature_engineered_data):
    """
    test perform_feature_engineering
    """
    X_train, _, y_train, _ = feature_engineered_data

    assert len(X_train) + len(y_train) > len(
        X_train), logging.error('No columns were created ' 'in F.E.')
    logging.info('test_perform_feature_engineering was successful')


def test_train_models(feature_engineered_data):
    """
    test train_models
    """
    X_train, X_test, y_train, y_test = feature_engineered_data

    train_models(X_train, X_test, y_train, y_test)
    rfc_model = joblib.load('./models/rfc_model.pkl')

    y_test_preds_rf = rfc_model.predict(X_test)
    acc = accuracy_score(y_test, y_test_preds_rf)
    assert acc >= 0.7, logging.error('Accuracy is lower than fixed threshold')
    logging.info('test_train_models was successful')


if __name__ == "__main__":
    pass
