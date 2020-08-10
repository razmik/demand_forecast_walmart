"""
Author: Rashmika Nawaratne
Date: 08-Aug-20 at 9:13 PM 
"""
import pandas as pd
from os import listdir
from os.path import join

LGBM_ROOT = 'model_outputs/lgbm_per_store_v2_store_wise/'
NN_ROOT = 'model_outputs/nn_per_store_v2_store_wise/'


def get_accuracy_df(filename):
    eval_data = []
    for fol in listdir(filename):
        for fl in listdir(join(filename, fol)):
            if 'evaluation.csv' in fl:
                store_id = fol.split('_')[-1]
                df = pd.read_csv(join(filename, fol, fl))
                mse = df.loc[(df['Data split'] == 'Test') & (df['Target unit'] == 'Unit')].iloc[0]['MSE']
                RMSE = df.loc[(df['Data split'] == 'Test') & (df['Target unit'] == 'Unit')].iloc[0]['RMSE']
                MAE = df.loc[(df['Data split'] == 'Test') & (df['Target unit'] == 'Unit')].iloc[0]['MAE']
                NWRMSLE = df.loc[(df['Data split'] == 'Test') & (df['Target unit'] == 'Log')].iloc[0]['MAE']
                row = [store_id, mse, RMSE, MAE, NWRMSLE]
                eval_data.append(row)
    return pd.DataFrame(eval_data, columns=['store', 'mse', 'rmse', 'mae', 'NWRMSLE'])


if __name__ == "__main__":

    lgbm_df = get_accuracy_df(LGBM_ROOT)
    # nn_df = get_accuracy_df(NN_ROOT)

    # print(nn_df.mse.mean(), nn_df.rmse.mean(), nn_df.mae.mean())
    print(lgbm_df.mse.mean(), lgbm_df.rmse.mean(), lgbm_df.mae.mean(), lgbm_df.NWRMSLE.mean())

    print('test')
