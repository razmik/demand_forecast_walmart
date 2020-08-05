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

from xgboost import XGBRegressor
from xgboost import Booster
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from favorita.load_data import Data


MODEL_NAME = 'base_xgb'
OUTPUT_FOLDER = 'model_outputs/' + MODEL_NAME
SELECTED_STORES = [i for i in range(1, 5)]
ONLY_EVALUATE = True


if __name__ == "__main__":

    start_time = time.time()
    data = Data()
    end_time = time.time()
    print("Load data in: {} mins.".format((end_time - start_time) / 60))

    # Filter stores to reduce the dataset
    data.train = data.train.loc[data.train.store_nbr.isin(SELECTED_STORES)]

    # Feature Engineering
    data.train['month'] = data.train['date'].dt.month
    data.train['week'] = data.train['date'].dt.week
    data.train['day'] = data.train['date'].dt.dayofweek

    data.train['month'] = data.train['month'].astype('int8')
    data.train['week'] = data.train['week'].astype('int8')
    data.train['day'] = data.train['day'].astype('int8')

    # Log transform the target variable (unit_sales)
    data.train['unit_sales'] = data.train['unit_sales'].apply(lambda u: np.log1p(float(u)) if float(u) > 0 else 0)

    # Merge tables
    df_full = pd.merge(data.train, data.items[['item_nbr', 'perishable', 'family']],
                       on='item_nbr')  # Train and items (perishable state)
    df_full = pd.merge(df_full,
                       data.weather_oil_holiday[['date', 'store_nbr', 'is_holiday', 'AvgTemp', 'dcoilwtico_imputed']],
                       on=['date', 'store_nbr'], how='left')  # Merge weather, oil and holiday

    del df_full['id']
    df_full.rename(columns={'dcoilwtico_imputed': 'oil_price', 'AvgTemp': 'avg_temp'}, inplace=True)

    # Get test train split
    df_train = df_full[df_full['date'] < datetime(2017, 7, 1)]
    df_test = df_full[df_full['date'] >= datetime(2017, 7, 1)]

    # clean variables
    del data
    del df_full
    gc.collect()

    # Modeling
    feature_columns = ['store_nbr', 'item_nbr', 'onpromotion', 'month', 'week', 'day', 'perishable', 'is_holiday',
                       'avg_temp', 'oil_price']
    target_column = ['unit_sales']

    X_train, Y_train = df_train[feature_columns], df_train[target_column]
    X_test, Y_test = df_test[feature_columns], df_test[target_column]
    print('Training dataset: {}'.format(X_train.shape))
    print('Testing dataset: {}'.format(X_test.shape))

    if not ONLY_EVALUATE:

        # Default XGB
        model_xgr_1 = XGBRegressor()
        start_time = time.time()
        model_xgr_1.fit(X_train, Y_train)
        end_time = time.time()
        print("Model Train time: {} mins.".format((end_time - start_time) / 60))

        # Save model
        model_xgr_1._Booster.save_model(OUTPUT_FOLDER + '.model')

    else:
        # Load from file
        model_xgr_1 = Booster().load_model(OUTPUT_FOLDER + '.model')

    Y_pred = model_xgr_1.predict(X_test)

    # Get target variables back from log (antilog)
    Y_pred_antilog = np.clip(np.expm1(Y_pred), 0, 1000)
    Y_test_antilog = np.expm1(Y_test)

    # Evaluation
    mse = mean_squared_error(Y_test_antilog, Y_pred_antilog)
    mae = mean_absolute_error(Y_test_antilog, Y_pred_antilog)
    rmse = np.sqrt(mse)
    print('RMSE: {}\nMSE: {}\nMAE: {}'.format(rmse, mse, mae))

    # Visualize
    plt.figure()

    plt.scatter(Y_test_antilog, Y_pred_antilog, color='blue')
    plt.xlabel("Unit Sales")
    plt.ylabel("Predicted Unit Sales")
    plt.title("Actual vs Predicted Unit Sales")
    plt.show()
