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

# Modeling and Feature Engineering
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Custom Files
from Evaluation import Evaluation


# Gloabl Params
DATA_FOLDER = 'data/Walmart_data_cleaned.csv'
PKL_MODEL_FILENAME = "model_outputs/rf_global_grid_search_model.pkl"

if __name__ == "__main__":

    df_orig = pd.read_csv(DATA_FOLDER)

    # Hyper parameters
    # Number of trees in random forest
    n_estimators = [79, 80, 81, 82, 85]
    max_features = ['auto']
    max_depth = [25, 27, 28, 30]
    min_samples_split = [2]
    min_samples_leaf = [1]
    bootstrap = [True]
    # Create the random grid
    search_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    print(search_grid)

    # Forming Data
    df_X = df_orig[['Store', 'Dept', 'IsHoliday', 'Type', 'Size', 'CPI', 'Unemployment', 'Month', 'Week']]
    df_Y = df_orig['Weekly_Sales']

    X_train, X_test, Y_train, Y_test = train_test_split(df_X, df_Y, test_size=0.3, random_state=8)

    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_grid = GridSearchCV(estimator=rf, param_grid=search_grid, cv=3, verbose=2, n_jobs=-1)
    # Fit the random search model
    rf_grid.fit(X_train, Y_train)

    # Persist Model
    # Save to file in the current working directory

    with open(PKL_MODEL_FILENAME, 'wb') as file:
        pickle.dump(rf_grid, file)

    # Load from file
    with open(PKL_MODEL_FILENAME, 'rb') as file:
        pickle_model = pickle.load(file)

    best_search = pickle_model.best_estimator_
    best_search_params = pickle_model.best_params_
    grid_accuracy = Evaluation.evaluate(best_search, X_test, Y_test)

    print(best_search_params)


