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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

# Custom Files
from Evaluation import Evaluation

# Gloabl Params
DATA_FOLDER = 'data/Walmart_data_cleaned.csv'
PKL_MODEL_FILENAME = "model_outputs/rf_global_model.pkl"


if __name__ == "__main__":

    df_orig = pd.read_csv(DATA_FOLDER)

    # Hyper parameters
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=40, stop=100, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(20, 50, num=6)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 6]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    print(random_grid)

    # Forming Data
    df_X = df_orig[['Store', 'Dept', 'IsHoliday', 'Type', 'Size', 'CPI', 'Unemployment', 'Month', 'Week']]
    df_Y = df_orig['Weekly_Sales']

    X_train, X_test, Y_train, Y_test = train_test_split(df_X, df_Y, test_size=0.3, random_state=8)

    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=8, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(X_train, Y_train)

    # Persist Model
    # Save to file in the current working directory

    with open(PKL_MODEL_FILENAME, 'wb') as file:
        pickle.dump(rf_random, file)

    # Load from file
    with open(PKL_MODEL_FILENAME, 'rb') as file:
        pickle_model = pickle.load(file)

    best_random = pickle_model.best_estimator_
    random_accuracy = Evaluation.evaluate(best_random, X_test, Y_test)
