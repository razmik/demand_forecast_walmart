"""
Author: Rashmika Nawaratne
Date: 31-Jul-20 at 1:13 PM 
"""
import numpy as  np
from sklearn.metrics import mean_squared_error


class Evaluation:

    @staticmethod
    def evaluate(model, x_test, y_test):

        predicted_values = model.predict(x_test)

        weights = x_test.IsHoliday.apply(lambda x: 5 if x else 1)
        wmae = np.round(np.sum(weights * abs(y_test - predicted_values)) / (np.sum(weights)), 2)

        rmse = np.sqrt(mean_squared_error(y_test, predicted_values))

        print('RMSE: {}, WMAE: {}'.format(rmse, wmae))

        return wmae, rmse
