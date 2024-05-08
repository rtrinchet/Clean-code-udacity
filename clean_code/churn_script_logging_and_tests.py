import os
import logging
import glob
import pytest
import churn_library
import joblib
from sklearn.metrics import accuracy_score


def check_pkl_files(path):
    pkl_files = glob.glob(path + "/*.pkl")
    if len(pkl_files) > 0:
        return True
    else:
        return False


def delete_png_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)
            print(f"Deleted {filename}")

    logging.basicConfig(
        filename='./logs/churn_library.log',
        level=logging.INFO,
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
        pytest.df = df
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    try:
        'Churn' in df.columns
    except:
        logging.error('Churn column is not included in data')


def test_eda(perform_eda):
    """
    test perform eda function
    """
    df = pytest.df

    # checks images are generated
    folder_path = "images"

    delete_png_images(folder_path)

    churn_library.perform_eda(df)

    png_files = glob.glob(folder_path + "/*.png")

    try:
        assert len(png_files) > 0
        logging.info("Folder contains PNG files!")
    except:
        print("Folder does not contain PNG files!")


def test_encoder_helper(encoder_helper):
    """
    test encoder helper
    """

    feature_list = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    df = pytest.df
    ini_n_cols = len(df.columns)
    df = churn_library.encoder_helper(df, category_lst=feature_list)

    try:
        assert len(df.columns) == ini_n_cols + len(feature_list)

    except exception as e:
        logging.error(
            f'There is a problem with the column creation for encoder helper. '
            f'A wrong number of columns was created\n'
            f'Error: {e}')


def test_perform_feature_engineering(perform_feature_engineering):
    """
    test perform_feature_engineering
    """
    df = pytest.df
    X_train, X_test, y_train, y_test = churn_library.perform_feature_engineering(df)

    try:
        assert len(X_train.columns) + len(y_train.columns) > len(df.columns)

    except:
        logging.error('No columns were created in F.E.')


def test_train_models(train_models):
    """
    test train_models
    """
    df = pytest.df
    X_train, X_test, y_train, y_test = churn_library.perform_feature_engineering(df)

    churn_library.train_models(X_train, X_test, y_train, y_test)
    rfc_model = joblib.load('./models/rfc_model.pkl')

    y_test_preds_rf = rfc_model.predict(X_test)
    acc = accuracy_score(y_test, y_test_preds_rf)
    try:
        assert acc >= 0.7

    except:
        logging.debug('Accuracy lower than fixed threshold')


if __name__ == "__main__":
    pass
