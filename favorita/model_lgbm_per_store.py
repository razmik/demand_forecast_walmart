"""
Author: Rashmika Nawaratne
Date: 08-Aug-20 at 12:45 PM 
"""
import pandas as pd
import numpy as np
from datetime import datetime, date

import time
import pickle
import os
import gc

import lightgbm as lgb

import matplotlib.pyplot as plt

from favorita.load_data import Data
from favorita.feature_extractor import Features
from favorita.evaluation import Evaluator

# Algorithmic and data config params
DATA_BEGIN_INDICATOR = datetime(2017, 1, 1)
TRAIN_DATE = date(2017, 5, 24)
VALIDATE_DATE = date(2017, 7, 12)
TEST_DATE = date(2017, 7, 31)
PERIODS_TO_COMBINE = 6
PREDICT_AHEAD_DAYS = 16

SELECTED_STORES = [i for i in range(1, 55)]
LOG_SCALED = True

# File config
VERSION = 2
MODEL_NAME = 'lgbm_per_store'
OUTPUT_ROOT = 'model_outputs/{}_v{}_store_wise/'.format(MODEL_NAME, VERSION)
if not os.path.exists(OUTPUT_ROOT):
    os.makedirs(OUTPUT_ROOT)

if __name__ == "__main__":

    print('Loading Data...')
    start_time = time.time()
    data = Data(test_the_script=False)
    print("Load data in: {} mins.".format((time.time() - start_time) / 60))

    # Loop through each store to develop a store wise model
    for selected_store in SELECTED_STORES:

        print('===' * 50)
        print('Store ', selected_store)
        print('===' * 50)

        OUTPUT_FOLDER = OUTPUT_ROOT + 'store_{}/'.format(selected_store)
        OUTPUT_FILENAME = OUTPUT_FOLDER + MODEL_NAME + '_{}'.format(selected_store)
        FIGURE_OUTPUT_FOLDER = OUTPUT_FOLDER + 'store_predictions/'
        FIGURE_OUTPUT_FILENAME = FIGURE_OUTPUT_FOLDER + 'store'

        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
        if not os.path.exists(FIGURE_OUTPUT_FOLDER):
            os.makedirs(FIGURE_OUTPUT_FOLDER)

        # Compose Feature Data
        print('Extracting Features...')
        try:
            start_time = time.time()
            fe = Features(data, selected_stores=[selected_store])
            feature_dictionary = fe.engineer_advanced_features(
                weeks_to_combine=PERIODS_TO_COMBINE, predict_periods=PREDICT_AHEAD_DAYS, train_date=TRAIN_DATE,
                valid_date=VALIDATE_DATE, test_date=TEST_DATE, log_scaled=LOG_SCALED, data_begin_indicator=DATA_BEGIN_INDICATOR)
        except:
            print('ERROR IN STORE {}'.format(selected_store))
            continue
        print("Features Composed in: {} mins.".format((time.time() - start_time) / 60))

        X_train, Y_train = feature_dictionary['train']
        X_val, Y_val = feature_dictionary['validation']
        X_test, Y_test = feature_dictionary['test']
        df_items_2017 = feature_dictionary['items_data']
        df_2017 = feature_dictionary['orig_data']

        # Predictive modeling
        print('Modeling...')
        params = {
            'num_leaves': 60,
            'objective': 'regression_l2',
            'max_depth': 10,
            'min_data_per_leaf': 300,
            'learning_rate': 0.05,
            'boosting_type': 'gbdt',
            'colsample_bytree': 0.4,
            'feature_fraction': 0.75,
            'bagging_fraction': 0.75,
            'bagging_freq': 1,
            'metric': 'l2',
            'num_threads': 4
        }

        MAX_ROUNDS = 1000
        val_pred = []
        test_pred = []
        cate_vars = []
        feature_importance_data = []

        for i in range(PREDICT_AHEAD_DAYS):

            print("=" * 50)
            print("Step %d" % (i + 1))
            print("=" * 50)

            dtrain = lgb.Dataset(
                X_train, label=Y_train[:, i],
                categorical_feature=cate_vars,
                weight=pd.concat([df_items_2017["perishable"]] * PERIODS_TO_COMBINE) * 0.25 + 1
            )
            dval = lgb.Dataset(
                X_val, label=Y_val[:, i], reference=dtrain,
                weight=df_items_2017["perishable"].values * 0.25 + 1,
                categorical_feature=cate_vars)
            bst = lgb.train(
                params, dtrain, num_boost_round=MAX_ROUNDS,
                valid_sets=[dtrain, dval], early_stopping_rounds=50, verbose_eval=50
            )

            f_importance = ["Step {}".format(i)] + [fx for fx in bst.feature_importance("gain")]
            feature_importance_data.append(f_importance)

            val_pred.append(bst.predict(
                X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))
            test_pred.append(bst.predict(
                X_test, num_iteration=bst.best_iteration or MAX_ROUNDS))

            # Save model
            with open(OUTPUT_FILENAME + '_model_step_{}.pickle'.format(i), 'wb') as file:
                pickle.dump(bst, file)

        print('Modeling completed.')

        # Save feature importance in a dataframe
        pd.DataFrame(feature_importance_data, columns=['Step'] + list(X_train.columns)).to_csv(OUTPUT_FILENAME + '_feature_importance.csv', index=False)

        # Evaluation

        weights = df_items_2017["perishable"].values * 0.25 + 1

        # If log scaled, get error for both log and anti-log
        eval = Evaluator()
        error_data = []
        columns = ['Target unit', 'Data split', 'MSE', 'RMSE', 'NWRMSLE', 'MAE', 'MAPE']
        if LOG_SCALED:
            mse_val_lg, rmse_val_lg, nwrmsle_val_lg, mae_val_lg, mape_val_lg = eval.get_error(weights, Y_val, np.array(val_pred).transpose(), PREDICT_AHEAD_DAYS)
            mse_val, rmse_val, nwrmsle_val, mae_val, mape_val = eval.get_error(weights, np.expm1(Y_val), np.clip(np.expm1(np.array(val_pred).transpose()), 0, 1000), PREDICT_AHEAD_DAYS)
            error_data.append(['Log', 'Validation', mse_val_lg, rmse_val_lg, nwrmsle_val_lg, mae_val_lg, mape_val_lg])
            error_data.append(['Unit', 'Validation', mse_val, rmse_val, nwrmsle_val, mae_val, mape_val])

            mse_test_lg, rmse_test_lg, nwrmsle_test_lg, mae_test_lg, mape_test_lg = eval.get_error(weights, Y_test, np.array(test_pred).transpose(), PREDICT_AHEAD_DAYS)
            mse_test, rmse_test, nwrmsle_test, mae_test, mape_test = eval.get_error(weights, np.expm1(Y_test), np.clip(np.expm1(np.array(test_pred).transpose()), 0, 1000), PREDICT_AHEAD_DAYS)
            error_data.append(['Log', 'Test', mse_test_lg, rmse_test_lg, nwrmsle_test_lg, mae_test_lg, mape_test_lg])
            error_data.append(['Unit', 'Test', mse_test, rmse_test, nwrmsle_test, mae_test, mape_test])
        else:
            mse_val, rmse_val, nwrmsle_val, mae_val, mape_val = eval.get_error(weights, Y_val, np.array(val_pred).transpose(), PREDICT_AHEAD_DAYS)
            mse_test, rmse_test, nwrmsle_test, mae_test, mape_test = eval.get_error(weights, Y_test, np.array(test_pred).transpose(), PREDICT_AHEAD_DAYS)
            error_data.append(['Unit', 'Validation', mse_val, rmse_val, nwrmsle_val, mae_val, mape_val])
            error_data.append(['Unit', 'Test', mse_test, rmse_test, nwrmsle_test, mae_test, mape_test])
        pd.DataFrame(error_data, columns=columns).to_csv(OUTPUT_FILENAME + '_evaluation.csv', index=False)

        # Model Output
        print("Combine with the test data split...")

        df_preds = pd.DataFrame(
            np.array(test_pred).transpose(), index=df_2017.index,
            columns=pd.date_range(TEST_DATE, periods=PREDICT_AHEAD_DAYS)
        ).stack().to_frame("log_predicted_unit_sales" if LOG_SCALED else "predicted_unit_sales")
        df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

        df_test = pd.DataFrame(
            Y_test, index=df_2017.index,
            columns=pd.date_range(TEST_DATE, periods=PREDICT_AHEAD_DAYS)
        ).stack().to_frame("log_actual_unit_sales" if LOG_SCALED else "actual_unit_sales")
        df_test.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

        comb_df = pd.concat([df_preds, df_test], axis=1)

        if LOG_SCALED:
            comb_df['predicted_unit_sales'] = np.clip(np.expm1(comb_df["log_predicted_unit_sales"]), 0, 1000)
            comb_df['actual_unit_sales'] = np.clip(np.expm1(comb_df["log_actual_unit_sales"]), 0, 1000)

        comb_merge_df = pd.merge(comb_df.reset_index(), df_items_2017.reset_index(), on='item_nbr')
        comb_merge_df.to_csv(OUTPUT_FILENAME + '_prediction_output.csv', index=False)

        # Plot for all stores
        groued_df = comb_df[['actual_unit_sales', 'predicted_unit_sales']].reset_index().groupby(
            ['date', 'store_nbr']).sum().reset_index()

        plt.figure(figsize=(10, 5))
        groued_df[groued_df.store_nbr == selected_store].plot(x="date", y=["actual_unit_sales", "predicted_unit_sales"])
        try:
            plt.savefig(FIGURE_OUTPUT_FILENAME + '_{}.png'.format(selected_store), dpi=300)
        except:
            plt.savefig(FIGURE_OUTPUT_FILENAME + '_{}.jpg'.format(selected_store), dpi=300)
        plt.clf()
        plt.close()

        del fe
        del feature_dictionary
        del df_2017
        del df_items_2017
        del df_test
        del df_preds
        del comb_df
        gc.collect()

    print('ALL COMPLETED')



