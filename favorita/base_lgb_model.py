"""
Author: Rashmika Nawaratne
Date: 05-Aug-20 at 4:53 PM 
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import time
import gc

import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from favorita.load_data import Data
from favorita.feature_extractor import Features

MODEL_NAME = 'base_lgb_v4'
OUTPUT_FOLDER = 'model_outputs/' + MODEL_NAME
SELECTED_STORES = [i for i in range(1, 60)]
ONLY_EVALUATE = False
TRAIN_TEST_SPLIT_DATE = datetime(2017, 7, 1)

if __name__ == "__main__":

    # Load data
    start_time = time.time()
    data = Data(test_the_script=False)
    print("Load data in: {} mins.".format((time.time() - start_time) / 60))

    # Filter stores to reduce the dataset
    data.train = data.train.loc[data.train.store_nbr.isin(SELECTED_STORES)]

    # Compose Feature Data
    start_time = time.time()
    feature_extractor = Features(data, TRAIN_TEST_SPLIT_DATE)
    print("Features Composed in: {} mins.".format((time.time() - start_time) / 60))

    df_train = feature_extractor.df_train
    df_validation = feature_extractor.df_validation
    df_test = feature_extractor.df_test

    # Modeling
    feature_columns = ['store_nbr', 'item_nbr', 'onpromotion', 'month', 'week', 'day', 'perishable', 'is_holiday',
                       'avg_temp', 'oil_price', 'family', 'item_class']
    categorical_columns = ['family', 'month', 'week', 'day', 'item_class']
    target_column = ['unit_sales']

    X_train, Y_train = df_train[feature_columns], df_train[target_column]
    X_val, Y_val = df_validation[feature_columns], df_validation[target_column]
    X_test, Y_test = df_test[feature_columns], df_test[target_column]
    print('Training dataset: {}'.format(X_train.shape))
    print('Testing dataset: {}'.format(X_test.shape))

    if not ONLY_EVALUATE:

        # converting datasets into lgb format,
        # list of names of categorical variable has been provided to conduct One-hot encoding
        lgb_train = lgb.Dataset(data=X_train, label=Y_train, categorical_feature=categorical_columns,
                                weight=X_train['perishable'] * 0.25 + 1)
        lgb_validation = lgb.Dataset(data=X_val, label=Y_val, categorical_feature=categorical_columns,
                               weight=X_val['perishable'] * 0.25 + 1, reference=lgb_train)

        # Parameters
        lgb_params = {'learning_rate': 0.1,
                      'metric': 'rmse',
                      'boosting_type': 'gbdt',
                      'max_depth': 10,
                      'num_leaves': 60,
                      'min_split_gain': 0.1,
                      'objective': 'regression',
                      'min_data_per_leaf': 300,
                      'colsample_bytree': 0.4,
                      'bagging_fraction': 0.9,
                      'num_threads': 4}

        eval_result = {}
        start_time = time.time()
        lgb_model = lgb.train(lgb_params,
                              num_boost_round=1000,
                              train_set=lgb_train,
                              valid_sets=[lgb_train, lgb_validation],
                              verbose_eval=True,
                              evals_result=eval_result,
                              early_stopping_rounds=10,
                              )
        end_time = time.time()
        print("Model Train time: {} mins.".format((end_time - start_time) / 60))

        # Save model
        with open(OUTPUT_FOLDER + '.pickle', 'wb') as file:
            pickle.dump(lgb_model, file)

    else:
        # Load from file
        with open(OUTPUT_FOLDER + '.pickle', 'rb') as file:
            lgb_model = pickle.load(file)

    Y_pred = lgb_model.predict(X_test)

    # Get target variables back from log (antilog)
    Y_pred_antilog = np.clip(np.expm1(Y_pred), 0, 1000)
    Y_test_antilog = np.expm1(Y_test)

    # Evaluation
    mse = mean_squared_error(Y_test_antilog, Y_pred_antilog, sample_weight=X_test['perishable'] * 0.25 + 1)
    mae = mean_absolute_error(Y_test_antilog, Y_pred_antilog, sample_weight=X_test['perishable'] * 0.25 + 1)
    rmse = np.sqrt(mse)
    print('RMSE: {}\nMSE: {}\nMAE: {}'.format(rmse, mse, mae))

    # Saving the feature importance results
    pd_feature_importance = pd.concat([pd.Series(lgb_model.feature_name(), name='Feature'),
                                       pd.Series(lgb_model.feature_importance(importance_type='split'), name='Split'),
                                       pd.Series(lgb_model.feature_importance(importance_type='gain'), name='Gain')],
                                      axis=1)
    pd_feature_importance.set_index('Feature', inplace=True)
    pd_feature_importance.to_csv(OUTPUT_FOLDER + '_feature_importance.csv')

    # Visualize
    plt.figure()

    plt.scatter(Y_test_antilog, Y_pred_antilog, color='blue')
    plt.xlabel("Unit Sales")
    plt.ylabel("Predicted Unit Sales")
    plt.title("Actual vs Predicted Unit Sales")
    plt.show()
