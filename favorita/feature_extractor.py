"""
Author: Rashmika Nawaratne
Date: 06-Aug-20 at 9:48 AM 
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta, date
import gc
from sklearn.preprocessing import LabelEncoder, StandardScaler


class Features(object):

    def __init__(self, data, selected_stores, train_test_split_date=datetime(2017, 1, 1)):
        self.data = data
        self.train_test_split_date = train_test_split_date
        self.selected_stores = selected_stores
        self.df_train = None
        self.df_validation = None
        self.df_test = None

    def engineer_base_features(self):

        """Setting Up Basic Features"""

        self.data.train['month'] = self.data.train['date'].dt.month
        self.data.train['week'] = self.data.train['date'].dt.week
        self.data.train['day'] = self.data.train['date'].dt.dayofweek

        self.data.train['month'] = self.data.train['month'].astype('int8')
        self.data.train['week'] = self.data.train['week'].astype('int8')
        self.data.train['day'] = self.data.train['day'].astype('int8')

        # Log transform the target variable (unit_sales)
        self.data.train['unit_sales'] = self.data.train['unit_sales'].apply(
            lambda u: np.log1p(float(u)) if float(u) > 0 else 0)

        # Merge tables
        df_full = pd.merge(self.data.train, self.data.items[['item_nbr', 'perishable', 'family', 'class']],
                           on='item_nbr')  # Train and items (perishable state)
        df_full = pd.merge(df_full,
                           self.data.weather_oil_holiday[
                               ['date', 'store_nbr', 'is_holiday', 'AvgTemp', 'dcoilwtico_imputed']],
                           on=['date', 'store_nbr'], how='left')  # Merge weather, oil and holiday

        del df_full['id']
        df_full.rename(columns={'dcoilwtico_imputed': 'oil_price', 'AvgTemp': 'avg_temp', 'class': 'item_class'},
                       inplace=True)

        # Get test train split
        self.df_train = df_full[df_full['date'] < (self.train_test_split_date - timedelta(days=7 * 2))]
        self.df_validation = df_full[(df_full['date'] >= (self.train_test_split_date - timedelta(days=7 * 2))) & (
                    df_full['date'] < self.train_test_split_date)]
        self.df_test = df_full[df_full['date'] >= self.train_test_split_date]

        # Encode Catagorical columns
        enc_family = LabelEncoder()
        self.df_train['family'] = enc_family.fit_transform(self.df_train['family'])
        self.df_validation['family'] = enc_family.transform(self.df_validation['family'])
        self.df_test['family'] = enc_family.transform(self.df_test['family'])

        # clean variables
        del self.data
        del df_full
        gc.collect()

    def engineer_advanced_features(self, weeks_to_combine=4, predict_periods=16, train_date=date(2017, 5, 24),
                                   valid_date=date(2017, 7, 12), test_date=date(2017, 7, 31),
                                   data_begin_indicator=datetime(2017, 1, 1), log_scaled=True):

        # Filter dataset
        df_train = self.data.train.loc[
            (self.data.train.store_nbr.isin(self.selected_stores)) & (self.data.train.date >= data_begin_indicator)]

        if log_scaled:
            self.data.train['unit_sales'] = self.data.train['unit_sales'].apply(
                lambda u: np.log1p(float(u)) if float(u) > 0 else 0)

        # Merge datasets
        self.data.stores['city'] = self.data.stores.city.str.lower()
        df_weather_store = pd.merge(self.data.stores[['store_nbr', 'city']],
                                    self.data.weather[['date', 'AvgTemp', 'city']], on='city')

        df_train_ext = pd.merge(df_train, self.data.items[['item_nbr', 'perishable', 'family', 'class']],
                                on='item_nbr')  # Train and items (perishable state)
        df_train_ext = pd.merge(df_train_ext,
                                self.data.weather_oil_holiday[
                                    ['date', 'store_nbr', 'is_holiday', 'AvgTemp', 'dcoilwtico_imputed']],
                                on=['date', 'store_nbr'], how='left')  # Merge weather, oil and holiday

        df_train_ext.rename(columns={'dcoilwtico_imputed': 'oil_price', 'AvgTemp': 'avg_temp', 'class': 'item_class'},
                            inplace=True)
        df_train_ext.set_index(["store_nbr", "item_nbr", "date"], inplace=True)
        del df_train_ext['id']
        gc.collect()

        # Create Datasets to prep train/test data
        df_2017 = df_train.set_index(["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(level=-1).fillna(0)
        df_2017.columns = df_2017.columns.get_level_values(1)

        promo_2017 = df_train.set_index(["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(level=-1).fillna(0)
        promo_2017.columns = promo_2017.columns.get_level_values(1)

        holiday_2017 = df_train_ext[["is_holiday"]].unstack(level=-1).fillna(0)
        holiday_2017.columns = holiday_2017.columns.get_level_values(1)

        df_temp_2017 = df_train_ext[["avg_temp"]].unstack(level=-1).fillna(0)
        df_temp_2017.columns = df_temp_2017.columns.get_level_values(1)

        for col_date in tqdm(df_temp_2017.columns.get_level_values(0)):
            for row_store in df_temp_2017.index.levels[0]:
                avg_temp = \
                    df_weather_store.loc[
                        (df_weather_store.store_nbr == row_store) & (df_weather_store.date == col_date)][
                        'AvgTemp'].iloc[0]
                df_temp_2017.loc[df_temp_2017.index.get_level_values('store_nbr') == row_store, col_date] = avg_temp

        df_items = self.data.items.set_index('item_nbr')
        df_items_2017 = df_items.reindex(df_2017.index.get_level_values(1))

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
                X['mean_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28 - i, 4, freq='7D').mean(
                    axis=1).values
                X['mean_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140 - i, 20, freq='7D').mean(
                    axis=1).values
            for i in range(predict_periods):
                X["promo_{}".format(i)] = promo_2017[t2017 + timedelta(days=i)].values.astype(np.uint8)
                X["holiday_{}".format(i)] = holiday_2017[t2017 + timedelta(days=i)].values.astype(np.uint8)
            if is_train:
                y = df_2017[
                    pd.date_range(t2017, periods=predict_periods)
                ].values
                return X, y
            return X

        print("Preparing dataset...")
        X_l, y_l = [], []
        for i in range(weeks_to_combine):
            delta = timedelta(days=7 * i)
            X_tmp, y_tmp = prepare_dataset(train_date + delta)
            X_l.append(X_tmp)
            y_l.append(y_tmp)

        X_train = pd.concat(X_l, axis=0)
        y_train = np.concatenate(y_l, axis=0)
        X_val, y_val = prepare_dataset(valid_date)
        X_test, y_test = prepare_dataset(test_date)

        del X_l, y_l
        gc.collect()

        out_dict = {
            'train': (X_train, y_train),
            'validation': (X_val, y_val),
            'test': (X_test, y_test),
            'items_data': df_items_2017,
            'orig_data': df_2017
        }

        return out_dict

    def engineer_nn_features(self, weeks_to_combine=4, predict_periods=16, train_date=date(2017, 5, 24),
                             valid_date=date(2017, 7, 12), test_date=date(2017, 7, 31),
                             data_begin_indicator=datetime(2017, 1, 1), log_scaled=True):

        # Filter dataset
        df_train = self.data.train.loc[
            (self.data.train.store_nbr.isin(self.selected_stores)) & (self.data.train.date >= data_begin_indicator)]

        if log_scaled:
            self.data.train['unit_sales'] = self.data.train['unit_sales'].apply(
                lambda u: np.log1p(float(u)) if float(u) > 0 else 0)

        # Merge datasets
        self.data.stores['city'] = self.data.stores.city.str.lower()
        df_weather_store = pd.merge(self.data.stores[['store_nbr', 'city']],
                                    self.data.weather[['date', 'AvgTemp', 'city']], on='city')

        df_train_ext = pd.merge(df_train, self.data.items[['item_nbr', 'perishable', 'family', 'class']],
                                on='item_nbr')  # Train and items (perishable state)
        df_train_ext = pd.merge(df_train_ext,
                                self.data.weather_oil_holiday[
                                    ['date', 'store_nbr', 'is_holiday', 'AvgTemp', 'dcoilwtico_imputed']],
                                on=['date', 'store_nbr'], how='left')  # Merge weather, oil and holiday

        df_train_ext.rename(columns={'dcoilwtico_imputed': 'oil_price', 'AvgTemp': 'avg_temp', 'class': 'item_class'},
                            inplace=True)
        df_train_ext.set_index(["store_nbr", "item_nbr", "date"], inplace=True)
        del df_train_ext['id']
        gc.collect()

        # Encoding Stores data
        enc_family = LabelEncoder()
        enc_city = LabelEncoder()
        enc_state = LabelEncoder()
        enc_type = LabelEncoder()

        self.data.items['family'] = enc_family.fit_transform(self.data.items['family'].values)
        self.data.stores['city'] = enc_city.fit_transform(self.data.stores['city'].values)
        self.data.stores['state'] = enc_state.fit_transform(self.data.stores['state'].values)
        self.data.stores['type'] = enc_type.fit_transform(self.data.stores['type'].values)

        # Create Datasets to prep train/test data
        df_2017 = df_train.set_index(["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(level=-1).fillna(0)
        df_2017.columns = df_2017.columns.get_level_values(1)

        promo_2017 = df_train.set_index(["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(level=-1).fillna(0)
        promo_2017.columns = promo_2017.columns.get_level_values(1)

        holiday_2017 = df_train_ext[["is_holiday"]].unstack(level=-1).fillna(0)
        holiday_2017.columns = holiday_2017.columns.get_level_values(1)

        df_temp_2017 = df_train_ext[["avg_temp"]].unstack(level=-1).fillna(0)
        df_temp_2017.columns = df_temp_2017.columns.get_level_values(1)

        for col_date in tqdm(df_temp_2017.columns.get_level_values(0)):
            for row_store in df_temp_2017.index.levels[0]:
                avg_temp = \
                    df_weather_store.loc[
                        (df_weather_store.store_nbr == row_store) & (df_weather_store.date == col_date)][
                        'AvgTemp'].iloc[0]
                df_temp_2017.loc[df_temp_2017.index.get_level_values('store_nbr') == row_store, col_date] = avg_temp

        self.data.items = self.data.items.set_index('item_nbr')
        self.data.stores = self.data.stores.set_index('store_nbr')
        df_items_2017 = self.data.items.reindex(df_2017.index.get_level_values(1))
        df_stores_2017 = self.data.stores.reindex(df_2017.index.get_level_values(0))

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
                X['mean_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28 - i, 4, freq='7D').mean(
                    axis=1).values
                X['mean_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140 - i, 20, freq='7D').mean(
                    axis=1).values
            for i in range(predict_periods):
                X["promo_{}".format(i)] = promo_2017[t2017 + timedelta(days=i)].values.astype(np.uint8)
                X["holiday_{}".format(i)] = holiday_2017[t2017 + timedelta(days=i)].values.astype(np.uint8)
            if is_train:
                y = df_2017[
                    pd.date_range(t2017, periods=predict_periods)
                ].values
                return X, y
            return X

        print("Preparing dataset...")
        X_l, y_l = [], []
        for i in range(weeks_to_combine):
            delta = timedelta(days=7 * i)
            X_tmp, y_tmp = prepare_dataset(train_date + delta)
            X_tmp = pd.concat([X_tmp, df_items_2017.reset_index(), df_stores_2017.reset_index()], axis=1)

            X_l.append(X_tmp)
            y_l.append(y_tmp)

        X_train = pd.concat(X_l, axis=0)
        y_train = np.concatenate(y_l, axis=0)
        X_val, y_val = prepare_dataset(valid_date)
        X_val = pd.concat([X_val, df_items_2017.reset_index(), df_stores_2017.reset_index()], axis=1)

        X_test, y_test = prepare_dataset(test_date)
        X_test = pd.concat([X_test, df_items_2017.reset_index(), df_stores_2017.reset_index()], axis=1)

        del X_l, y_l
        gc.collect()

        scaler = StandardScaler()
        scaler.fit(pd.concat([X_train, X_val, X_test]))
        X_train[:] = scaler.transform(X_train)
        X_val[:] = scaler.transform(X_val)
        X_test[:] = scaler.transform(X_test)

        X_train = X_train.values
        X_test = X_test.values
        X_val = X_val.values
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

        out_dict = {
            'train': (X_train, y_train),
            'validation': (X_val, y_val),
            'test': (X_test, y_test),
            'items_data': df_items_2017,
            'orig_data': df_2017
        }

        return out_dict
