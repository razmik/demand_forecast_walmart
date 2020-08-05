"""
Author: Rashmika Nawaratne
Date: 31-Jul-20 at 12:06 PM

Approach:
Use Random Search to narrow down the range
Next use Grid Search to specify the optimal point

"""

# Data Wrangling and Viz
import pandas as pd
import pickle
import numpy as np

# Modeling and Feature Engineering
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Custom Files
from Evaluation import Evaluation


# Gloabl Params
DATA_FOLDER = 'data/Walmart_data_cleaned.csv'
PKL_MODEL_FILENAME = "model_outputs/xgb_global_grid_search_model.pkl"

if __name__ == "__main__":

    df_orig = pd.read_csv(DATA_FOLDER)

    # Hyper parameters
    param_grid = {
        'max_depth': [19, 20, 21, 22, 25, 30],
        'learning_rate': [0.15, 0.2, 0.3, 0.5, 0.8, 1],
        'subsample': [1.0],
        'colsample_bytree': [1.0],
        'colsample_bylevel': [1.0],
        'min_child_weight': [6.0, 6.5, 7.0, 7.5, 8],
        'gamma': [1.0],
        'reg_lambda': [0.9, 1.0, 1.3, 1.5, 2.0, 3.0],
        'n_estimators': [75, 78, 80, 81, 82, 83, 85]}

    # Forming Data
    df_X = df_orig[['Store', 'Dept', 'IsHoliday', 'Type', 'Size', 'CPI', 'Unemployment', 'Month', 'Week']]
    df_Y = df_orig['Weekly_Sales']

    X_train, X_test, Y_train, Y_test = train_test_split(df_X, df_Y, test_size=0.3, random_state=8)

    # First create the base model to tune
    xgr = xgb.XGBRegressor()

    # Random search of parameters, using 3 fold cross validation,
    xgr_grid = GridSearchCV(estimator=xgr, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)

    # Fit the random search model
    xgr_grid.fit(X_train, Y_train)

    # Persist Model
    # Save to file in the current working directory

    with open(PKL_MODEL_FILENAME, 'wb') as file:
        pickle.dump(xgr_grid, file)

    # Load from file
    with open(PKL_MODEL_FILENAME, 'rb') as file:
        pickle_model = pickle.load(file)

    best_search = pickle_model.best_estimator_
    best_search_params = pickle_model.best_params_
    grid_accuracy = Evaluation.evaluate(best_search, X_test, Y_test)

    print(best_search_params)


