"""
Author: Rashmika Nawaratne
Date: 09-Aug-20 at 11:58 AM 
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


class Evaluator(object):

    @staticmethod
    def get_error(weights, true_val, pred_val, predict_days_ahead):

        mse = mean_squared_error(true_val, pred_val, sample_weight=weights)
        rmse = np.sqrt(mse)

        mae = mean_absolute_error(true_val, pred_val, sample_weight=weights)

        mape = 0# np.mean(np.abs((true_val - pred_val) / true_val)) * 100

        try:
            err = (true_val - pred_val) ** 2
            err = err.sum(axis=1) * weights
            nwrmsle = np.sqrt(err.sum() / weights.sum() / predict_days_ahead)
        except:
            nwrmsle = rmse

        return mse, rmse, nwrmsle, mae, mape

