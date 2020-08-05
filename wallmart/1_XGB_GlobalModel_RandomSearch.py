"""
Author: Rashmika Nawaratne
Date: 31-Jul-20 at 12:06 PM

Approach:
Use Random Search to narrow down the range
Next use Grid Search to specify the optimal point

"""

# Data Wrangling and Viz
import pandas as pd
import numpy as np
import pickle

# Modeling and Feature Engineering
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

# Custom Files
from Evaluation import Evaluation

# Gloabl Params
DATA_FOLDER = 'data/Walmart_data_cleaned.csv'
PKL_MODEL_FILENAME = "model_outputs/xgr_global_model_random_search.pkl"

if __name__ == "__main__":
    df_orig = pd.read_csv(DATA_FOLDER)

    # Hyper parameters
    param_grid = {
        'max_depth': [6, 10, 15, 20],
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0, 3],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
        'gamma': [0, 0.25, 0.5, 1.0],
        'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
        'n_estimators': [int(x) for x in np.linspace(start=40, stop=100, num=10)]}

    # Forming Data
    df_X = df_orig[['Store', 'Dept', 'IsHoliday', 'Type', 'Size', 'CPI', 'Unemployment', 'Month', 'Week']]
    df_Y = df_orig['Weekly_Sales']

    X_train, X_test, Y_train, Y_test = train_test_split(df_X, df_Y, test_size=0.3, random_state=8)

    # First create the base model to tune
    xgr = xgb.XGBRegressor()

    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    xgr_random = RandomizedSearchCV(estimator=xgr, param_distributions=param_grid, n_iter=100, cv=3, verbose=2,
                                    random_state=8, n_jobs=-1)
    # Fit the random search model
    xgr_random.fit(X_train, Y_train)

    # Persist Model
    # Save to file in the current working directory

    with open(PKL_MODEL_FILENAME, 'wb') as file:
        pickle.dump(xgr_random, file)

    # Load from file
    with open(PKL_MODEL_FILENAME, 'rb') as file:
        pickle_model = pickle.load(file)

    best_random = pickle_model.best_estimator_
    best_random_params = pickle_model.best_params_
    random_accuracy = Evaluation.evaluate(best_random, X_test, Y_test)

    print(best_random_params)