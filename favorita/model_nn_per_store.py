"""
Author: Rashmika Nawaratne
Date: 08-Aug-20 at 12:45 PM 
"""
import pandas as pd
import numpy as np
from datetime import datetime, date

import time
import os
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt

from favorita.load_data import Data
from favorita.feature_extractor import Features

# Algorithmic and data config params
DATA_BEGIN_INDICATOR = datetime(2017, 1, 1)
TRAIN_DATE = date(2017, 5, 24)
VALIDATE_DATE = date(2017, 7, 12)
TEST_DATE = date(2017, 7, 31)
PERIODS_TO_COMBINE = 6
PREDICT_AHEAD_DAYS = 16
N_EPOCHS = 1000

SELECTED_STORES = [i for i in range(1, 20)]
LOG_SCALED = True

# File config
VERSION = 1
MODEL_NAME = 'nn_per_store'

if __name__ == "__main__":

    print('Loading Data...')
    start_time = time.time()
    data = Data(test_the_script=False)
    print("Load data in: {} mins.".format((time.time() - start_time) / 60))

    # Loop through each store to develop a store wise model
    for selected_store in SELECTED_STORES:

        print('\n\n\n===' * 50)
        print('Store ', selected_store)
        print('===' * 50)

        OUTPUT_FOLDER = 'model_outputs/{}_{}_store_{}/'.format(MODEL_NAME, VERSION, selected_store)
        OUTPUT_FILENAME = OUTPUT_FOLDER + MODEL_NAME
        FIGURE_OUTPUT_FOLDER = OUTPUT_FOLDER + 'store_predictions/'
        FIGURE_OUTPUT_FILENAME = FIGURE_OUTPUT_FOLDER + 'store'

        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
        if not os.path.exists(FIGURE_OUTPUT_FOLDER):
            os.makedirs(FIGURE_OUTPUT_FOLDER)

        # Compose Feature Data
        print('Extracting Features...')
        start_time = time.time()
        fe = Features(data, selected_stores=[selected_store])
        feature_dictionary = fe.engineer_nn_features(
            weeks_to_combine=PERIODS_TO_COMBINE, predict_periods=PREDICT_AHEAD_DAYS, train_date=TRAIN_DATE,
            valid_date=VALIDATE_DATE, test_date=TEST_DATE, log_scaled=LOG_SCALED, data_begin_indicator=DATA_BEGIN_INDICATOR)
        print("Features Composed in: {} mins.".format((time.time() - start_time) / 60))

        X_train, Y_train = feature_dictionary['train']
        X_val, Y_val = feature_dictionary['validation']
        X_test, Y_test = feature_dictionary['test']
        df_items_2017 = feature_dictionary['items_data']
        df_2017 = feature_dictionary['orig_data']

        # Predictive modeling
        print('Modeling...')

        # Model architecture
        def get_model_architecture():
            model = Sequential()

            model.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(BatchNormalization())
            model.add(Dropout(.1))

            model.add(Dense(32, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(.2))

            model.add(Dense(1))

            return model

        # Train model

        val_pred = []
        test_pred = []

        sample_weights = np.array(pd.concat([df_items_2017["perishable"]] * PERIODS_TO_COMBINE) * 0.25 + 1)

        for i in range(PREDICT_AHEAD_DAYS):
            print("=" * 50)
            print("Step %d" % (i + 1))
            print("=" * 50)

            y = Y_train[:, i]
            y_mean = y.mean()
            xv = X_val
            yv = Y_val[:, i]

            model = get_model_architecture()
            opt = optimizers.Adam(lr=0.001)
            model.compile(loss='mse', optimizer=opt, metrics=['mse'])

            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min')
            ]

            model.fit(X_train, y - y_mean, batch_size=65536, epochs=N_EPOCHS, verbose=2,
                      sample_weight=sample_weights, validation_data=(xv, yv - y_mean), callbacks=callbacks)

            val_pred.append(model.predict(X_val) + y_mean)
            test_pred.append(model.predict(X_test) + y_mean)

            model.save(OUTPUT_FILENAME + '_model_step_{}.model'.format(i))

        print('Modeling completed.')


        # Evaluation
        def get_error(weights, true_val, pred_val):

            err = (true_val - pred_val) ** 2
            err = err.sum(axis=1) * weights
            nwrmsle = np.sqrt(err.sum() / weights.sum() / PREDICT_AHEAD_DAYS)

            mse = mean_squared_error(true_val, pred_val, sample_weight=weights)
            rmse = np.sqrt(mse)

            mae = mean_absolute_error(true_val, pred_val, sample_weight=weights)

            return mse, rmse, nwrmsle, mae

        weights = df_items_2017["perishable"].values * 0.25 + 1

        # If log scaled, get error for both log and anti-log
        error_data = []
        columns = ['Target unit', 'Data split', 'MSE', 'RMSE', 'NWRMSLE', 'MAE']
        if LOG_SCALED:
            mse_val_lg, rmse_val_lg, nwrmsle_val_lg, mae_val_lg = get_error(weights, Y_val, np.array(val_pred).transpose()[0])
            mse_val, rmse_val, nwrmsle_val, mae_val = get_error(weights, np.expm1(Y_val), np.clip(np.expm1(np.array(val_pred).transpose()[0]), 0, 1000))
            error_data.append(['Log', 'Validation', mse_val_lg, rmse_val_lg, nwrmsle_val_lg, mae_val_lg])
            error_data.append(['Unit', 'Validation', mse_val, rmse_val, nwrmsle_val, mae_val])

            mse_test_lg, rmse_test_lg, nwrmsle_test_lg, mae_test_lg = get_error(weights, Y_test, np.array(test_pred).transpose()[0])
            mse_test, rmse_test, nwrmsle_test, mae_test = get_error(weights, np.expm1(Y_test), np.clip(np.expm1(np.array(test_pred).transpose()[0]), 0, 1000))
            error_data.append(['Log', 'Test', mse_test_lg, rmse_test_lg, nwrmsle_test_lg, mae_test_lg])
            error_data.append(['Unit', 'Test', mse_test, rmse_test, nwrmsle_test, mae_test])
        else:
            mse_val, rmse_val, nwrmsle_val, mae_val = get_error(weights, Y_val, np.array(val_pred).transpose()[0])
            mse_test, rmse_test, nwrmsle_test, mae_test = get_error(weights, Y_test, np.array(test_pred).transpose()[0])
            error_data.append(['Unit', 'Validation', mse_val, rmse_val, nwrmsle_val, mae_val])
            error_data.append(['Unit', 'Test', mse_test, rmse_test, nwrmsle_test, mae_test])
        pd.DataFrame(error_data, columns=columns).to_csv(OUTPUT_FILENAME + '_evaluation.csv', index=False)

        # Model Output
        print("Combine with the test data split...")

        df_preds = pd.DataFrame(
            np.array(test_pred).transpose()[0], index=df_2017.index,
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



