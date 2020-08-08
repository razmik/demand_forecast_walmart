"""LGBM Starter

This is watered-down version of one of my earlier scripts. 
Only very basic features are retained so hopefully it won't ruin the fun for you.
https://www.kaggle.com/ceshine/lgbm-starter/code
"""
from datetime import date, timedelta
import time
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from tqdm import tqdm
import gc
import os
import pickle

from favorita.load_data import Data

VERSION = 1
MODEL_NAME = 'starter_lgb'
OUTPUT_FOLDER = 'model_outputs/{}_{}/'.format(MODEL_NAME, VERSION)
OUTPUT_FILENAME = OUTPUT_FOLDER + MODEL_NAME
SELECTED_STORES = [i for i in range(1, 20)]
PERIODS_TO_COMBINE = 6

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

if __name__ == "__main__":

    start_time = time.time()
    data = Data(test_the_script=False)
    print("Load data in: {} mins.".format((time.time() - start_time) / 60))

    # Filter stores to reduce the dataset
    df_train = data.train.loc[data.train.store_nbr.isin(SELECTED_STORES)]
    df_items = data.items.set_index('item_nbr')

    print('Merging datasets...')
    data.stores['city'] = data.stores.city.str.lower()
    df_weather_store = pd.merge(data.stores[['store_nbr', 'city']], data.weather[['date', 'AvgTemp', 'city']],
                                on='city')

    df_train_ext = pd.merge(df_train, data.items[['item_nbr', 'perishable', 'family', 'class']],
                            on='item_nbr')  # Train and items (perishable state)
    df_train_ext = pd.merge(df_train_ext,
                            data.weather_oil_holiday[
                                ['date', 'store_nbr', 'is_holiday', 'AvgTemp', 'dcoilwtico_imputed']],
                            on=['date', 'store_nbr'], how='left')  # Merge weather, oil and holiday

    del df_train_ext['id']
    df_train_ext.rename(columns={'dcoilwtico_imputed': 'oil_price', 'AvgTemp': 'avg_temp', 'class': 'item_class'},
                        inplace=True)

    """Composing datasets for feature extraction"""

    df_2017 = df_train.set_index(["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(level=-1).fillna(0)
    df_2017.columns = df_2017.columns.get_level_values(1)

    promo_2017 = df_train.set_index(["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(level=-1).fillna(0)
    promo_2017.columns = promo_2017.columns.get_level_values(1)

    holiday_2017 = df_train_ext.set_index(["store_nbr", "item_nbr", "date"])[["is_holiday"]].unstack(level=-1).fillna(0)
    holiday_2017.columns = holiday_2017.columns.get_level_values(1)

    df_temp_2017 = df_train_ext.set_index(["store_nbr", "item_nbr", "date"])[["avg_temp"]].unstack(level=-1).fillna(0)
    df_temp_2017.columns = df_temp_2017.columns.get_level_values(1)
    for col_date in tqdm(df_temp_2017.columns.get_level_values(0)):
        for row_store in df_temp_2017.index.levels[0]:
            avg_temp = \
                df_weather_store.loc[(df_weather_store.store_nbr == row_store) & (df_weather_store.date == col_date)][
                    'AvgTemp'].iloc[0]
            df_temp_2017.loc[df_temp_2017.index.get_level_values('store_nbr') == row_store, col_date] = avg_temp

    df_items_2017 = df_items.reindex(df_2017.index.get_level_values(1))
    del df_items

    gc.collect()

    """Preparing the Dataset for modeling"""

    def get_timespan(df, dt, minus, periods, freq='D'):
        return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]


    def prepare_dataset(t2017, is_train=True):
        X = pd.DataFrame({
            "perishable": df_items_2017.perishable.values.ravel(),
            "day_1_2017": get_timespan(df_2017, t2017, 1, 1).values.ravel(),
            "mean_3_2017": get_timespan(df_2017, t2017, 3, 3).mean(axis=1).values,
            "mean_7_2017": get_timespan(df_2017, t2017, 7, 7).mean(axis=1).values,
            "mean_14_2017": get_timespan(df_2017, t2017, 14, 14).mean(axis=1).values,
            "mean_30_2017": get_timespan(df_2017, t2017, 30, 30).mean(axis=1).values,
            "mean_60_2017": get_timespan(df_2017, t2017, 60, 60).mean(axis=1).values,
            "mean_140_2017": get_timespan(df_2017, t2017, 140, 140).mean(axis=1).values,
            "promo_14_2017": get_timespan(promo_2017, t2017, 14, 14).sum(axis=1).values,
            "promo_60_2017": get_timespan(promo_2017, t2017, 60, 60).sum(axis=1).values,
            "promo_140_2017": get_timespan(promo_2017, t2017, 140, 140).sum(axis=1).values,
            "holiday_14_2017": get_timespan(holiday_2017, t2017, 14, 14).sum(axis=1).values,
            "holiday_60_2017": get_timespan(holiday_2017, t2017, 60, 60).sum(axis=1).values,
            "holiday_140_2017": get_timespan(holiday_2017, t2017, 140, 140).sum(axis=1).values,
            "day_1_temp": get_timespan(df_temp_2017, t2017, 1, 1).values.ravel(),
            "mean_2_temp": get_timespan(df_temp_2017, t2017, 2, 2).mean(axis=1).values,
            "mean_3_temp": get_timespan(df_temp_2017, t2017, 3, 3).mean(axis=1).values,
            "mean_5_temp": get_timespan(df_temp_2017, t2017, 5, 5).mean(axis=1).values,
            "mean_7_temp": get_timespan(df_temp_2017, t2017, 7, 7).mean(axis=1).values
        })
        for i in range(7):
            X['mean_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28 - i, 4, freq='7D').mean(axis=1).values
            X['mean_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140 - i, 20, freq='7D').mean(axis=1).values
        for i in range(16):
            X["promo_{}".format(i)] = promo_2017[t2017 + timedelta(days=i)].values.astype(np.uint8)
            # X["holiday_{}".format(i)] = holiday_2017[t2017 + timedelta(days=i)].values.astype(np.uint8)
        if is_train:
            y = df_2017[
                pd.date_range(t2017, periods=16)
            ].values
            return X, y

        return X


    print("Preparing dataset...")

    t2017 = date(2017, 5, 24)
    X_l, y_l = [], []
    for i in range(PERIODS_TO_COMBINE):
        delta = timedelta(days=7 * i)
        X_tmp, y_tmp = prepare_dataset(t2017 + delta)
        X_l.append(X_tmp)
        y_l.append(y_tmp)

    X_train = pd.concat(X_l, axis=0)
    y_train = np.concatenate(y_l, axis=0)
    del X_l, y_l
    X_val, y_val = prepare_dataset(date(2017, 7, 12))
    X_test, y_test = prepare_dataset(date(2017, 7, 31))
    gc.collect()

    """Training and predicting models"""

    print("Training and predicting models...")
    params = {
        'num_leaves': 2 ** 5 - 1,
        'objective': 'regression_l2',
        'max_depth': 8,
        'min_data_in_leaf': 50,
        'learning_rate': 0.05,
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

    for i in range(16):
        print("=" * 50)
        print("Step %d" % (i + 1))
        print("=" * 50)

        dtrain = lgb.Dataset(
            X_train, label=y_train[:, i],
            categorical_feature=cate_vars,
            weight=pd.concat([df_items_2017["perishable"]] * PERIODS_TO_COMBINE) * 0.25 + 1
        )
        dval = lgb.Dataset(
            X_val, label=y_val[:, i], reference=dtrain,
            weight=df_items_2017["perishable"].values * 0.25 + 1,
            categorical_feature=cate_vars)
        bst = lgb.train(
            params, dtrain, num_boost_round=MAX_ROUNDS,
            valid_sets=[dtrain, dval], early_stopping_rounds=50, verbose_eval=50
        )
        print("\n".join(("%s: %.2f" % x) for x in sorted(
            zip(X_train.columns, bst.feature_importance("gain")),
            key=lambda x: x[1], reverse=True
        )))
        val_pred.append(bst.predict(
            X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))
        test_pred.append(bst.predict(
            X_test, num_iteration=bst.best_iteration or MAX_ROUNDS))

        # Save model
        with open(OUTPUT_FILENAME + '_step_{}.pickle'.format(i), 'wb') as file:
            pickle.dump(bst, file)

    print("Validation mse:", mean_squared_error(
        y_val, np.array(val_pred).transpose(), sample_weight=(df_items_2017["perishable"].values * 0.25 + 1)))

    print("Validation mse:", mean_squared_error(
        y_test, np.array(test_pred).transpose(), sample_weight=(df_items_2017["perishable"].values * 0.25 + 1)))
