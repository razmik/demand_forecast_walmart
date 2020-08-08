"""
Author: Rashmika Nawaratne
Date: 05-Aug-20 at 4:53 PM

Categorical features in LGBM:
https://www.kaggle.com/c/home-credit-default-risk/discussion/58950
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import time

import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt

from favorita.load_data import Data
from favorita.feature_extractor import Features

MODEL_NAME = 'base_lgb_random_search'
OUTPUT_FOLDER = 'model_outputs/' + MODEL_NAME
SELECTED_STORES = [i for i in range(1, 11)]
ONLY_EVALUATE = False
TRAIN_TEST_SPLIT_DATE = datetime(2017, 7, 1)

if __name__ == "__main__":

    # Load data
    start_time = time.time()
    data = Data(test_the_script=True)
    print("Load data in: {} mins.".format((time.time() - start_time) / 60))

    # Filter stores to reduce the dataset
    data.train = data.train.loc[data.train.store_nbr.isin(SELECTED_STORES)]

    # Compose Feature Data
    start_time = time.time()
    feature_extractor = Features(data, TRAIN_TEST_SPLIT_DATE)
    print("Features Composed in: {} mins.".format((time.time() - start_time) / 60))

    df_train = feature_extractor.df_train
    df_test = feature_extractor.df_test

    # Modeling
    feature_columns = ['store_nbr', 'item_nbr', 'onpromotion', 'month', 'week', 'day', 'perishable', 'is_holiday',
                       'avg_temp', 'oil_price', 'family']
    target_column = ['unit_sales']

    X_train, Y_train = df_train[feature_columns], df_train[target_column]
    X_test, Y_test = df_test[feature_columns], df_test[target_column]
    print('Training dataset: {}'.format(X_train.shape))
    print('Testing dataset: {}'.format(X_test.shape))

    if not ONLY_EVALUATE:

        # converting datasets into lgb format,
        # list of names of categorical variable has been provided to conduct One-hot encoding
        lgb_train = lgb.Dataset(data=X_train, label=Y_train, categorical_feature=['family'])
        lgb_test = lgb.Dataset(data=X_test, label=Y_test, categorical_feature=['family'], reference=lgb_train)

        # Parameters
        grid_params = {'learning_rate': [0.001, 0.01, 0.1, 0.2, 0, 3],
                       'num_leaves': [50, 60, 80, 100],
                       'metric': ['rmse'],
                       'boosting_type': ['gbdt'],
                       'bagging_fraction': [0.8, 0.85, 0.9],
                       'max_depth': [6, 8, 10, 12],
                       'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                       'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                       'objective': ['regression'],
                       'min_data_per_leaf': [80, 100, 120, 150, 180, 210, 250, 300],
                       'min_split_gain': [0.01, 0.05, 0.1]}

        # training the model using 100 iterations with early stopping if validation RMSE decreases
        eval_result = {}
        start_time = time.time()
        lgb_random = RandomizedSearchCV(estimator=lgb.LGBMRegressor(), param_distributions=grid_params, n_iter=10, cv=5, verbose=2,
                                    random_state=8, n_jobs=-1)
        lgb_random.fit(X_train, Y_train)
        end_time = time.time()
        print("Model Grid search time: {} mins.".format((end_time - start_time) / 60))

        # Save model
        with open(OUTPUT_FOLDER + '.pickle', 'wb') as file:
            pickle.dump(lgb_random, file)

    else:
        # Load from file
        with open(OUTPUT_FOLDER + '.pickle', 'rb') as file:
            lgb_random = pickle.load(file)

    # Evaluation

    best_random = lgb_random.best_estimator_
    best_random_params = lgb_random.best_params_

    Y_pred = best_random.predict(X_test)

    # Get target variables back from log (antilog)
    Y_pred_antilog = np.clip(np.expm1(Y_pred), 0, 1000)
    Y_test_antilog = np.expm1(Y_test)

    # Evaluation
    mse = mean_squared_error(Y_test_antilog, Y_pred_antilog)
    mae = mean_absolute_error(Y_test_antilog, Y_pred_antilog)
    rmse = np.sqrt(mse)
    print('RMSE: {}\nMSE: {}\nMAE: {}\n'.format(rmse, mse, mae))

    print('Best Parameters:\n', best_random_params)

    # Visualize
    plt.figure()

    plt.scatter(Y_test_antilog, Y_pred_antilog, color='blue')
    plt.xlabel("Unit Sales")
    plt.ylabel("Predicted Unit Sales")
    plt.title("Actual vs Predicted Unit Sales")
    plt.savefig(OUTPUT_FOLDER + '.png', dpi=800)
