"""
Author: Rashmika Nawaratne
Date: 05-Aug-20 at 4:28 PM

Bootstrapped from Anton Mashikhin, 2017.
https://www.kaggle.com/heyt0ny/read-data-for-low-memory-usage
"""
import pandas as pd
import favorita.config as config


class Data(object):

    def __init__(self, test_the_script=False):
        self.DATA_FOLDER = config.ROOT_FOLDER
        self.test_the_script = test_the_script
        self.read_data()
        print('Train shape: ', self.train.shape)

    def read_data(self):

        self.nrows = None
        if self.test_the_script:
            self.nrows = 1000

        self.train = self.read_train_test_low_memory(train_flag=True)
        self.test = self.read_train_test_low_memory(train_flag=False)
        self.stores = self.read_stores_low_memory()
        self.items = self.read_items_low_memory()
        self.weather_oil_holiday = self.read_oil_weather_holiday_low_memory()
        self.transactions = self.read_transactions_low_memory()

    def read_train_test_low_memory(self, train_flag=True):
        filename = 'train'
        if not train_flag: filename = 'test'

        types = {'id': 'int64',
                 'item_nbr': 'int32',
                 'store_nbr': 'int8',
                 'unit_sales': 'float32'
                 }
        # Skipping rows till 2016
        data = pd.read_csv(self.DATA_FOLDER + filename + '.csv', parse_dates=['date'], dtype=types,
                           nrows=self.nrows, infer_datetime_format=True, low_memory=True, skiprows=range(1, 66458909))

        # Missing value imputation for onpromotion column
        data['onpromotion'].fillna(False, inplace=True)
        data['onpromotion'] = data['onpromotion'].map({False: 0, True: 1})
        data['onpromotion'] = data['onpromotion'].astype('int8')

        # Clip sales values 0-100
        if train_flag:
            data['unit_sales'].clip(0, 1000, inplace=True)

        return data

    def read_stores_low_memory(self):
        types = {'cluster': 'int32',
                 'store_nbr': 'int8',
                 }
        data = pd.read_csv(self.DATA_FOLDER + 'stores.csv', dtype=types, low_memory=True)
        return data

    def read_items_low_memory(self):
        types = {'item_nbr': 'int32',
                 'perishable': 'int8',
                 'class': 'int16'
                 }
        data = pd.read_csv(self.DATA_FOLDER + 'items.csv', dtype=types, low_memory=True)
        return data

    def read_oil_weather_holiday_low_memory(self):
        types = {'dcoilwtico_imputed': 'float32',
                 'AvgTemp': 'int8',
                 'store_nbr': 'int8',
                 'is_holiday': 'int8'
                 }
        data = pd.read_csv(self.DATA_FOLDER + 'holiday_weather_oil_combined.csv', parse_dates=['date'], dtype=types,
                           infer_datetime_format=True, low_memory=True)
        return data

    def read_transactions_low_memory(self):
        types = {'transactions': 'int16',
                 'store_nbr': 'int8'
                 }
        data = pd.read_csv(self.DATA_FOLDER + 'transactions.csv', parse_dates=['date'], dtype=types,
                           infer_datetime_format=True, low_memory=True)
        return data
