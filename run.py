import pickle
import pandas as pd
import numpy as np
import json
import time
import argparse

from framework import run_measurements, viz


# Don't forget to point the output to a log file

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--form', type=str, required=True, help="Data form, one of: ['frac_diff' , 'first_order_diff', 'original']")
    args = parser.parse_args()

    data_form = args.form.lower()

    valid_forms = ['frac_diff' , 'first_order_diff', 'original', 'ta_original', 'ta_fod']
    if data_form not in valid_forms:
        raise ValueError(f'The data_form arg must be one of: {valid_forms}')
    

    start = time.time()

    ####################################### Load Data ##############################################################################################

    with open('/mnt/c/Users/resha/Documents/Github/balancing_framework/spy5m_bintp_labelled.pkl', 'rb') as f:
        df_original = pickle.load(f)
    with open('/mnt/c/Users/resha/Documents/Github/balancing_framework/spy5m_bintp004_episodes_fracdiff.pkl', 'rb') as f:
        df_fd = pickle.load(f)
    with open('/mnt/c/Users/resha/Documents/Github/balancing_framework/spy5m_labelled_episodes_ta.pkl', 'rb') as f:
        df_ta = pickle.load(f)
        df_ta['label'] = df_original['tp_0.004'][df_ta.index]
    # PZ algorithm has some look ahead so remove the episode labels, will be uesd only for some kind of analysis afterwards
    # df = df_original.drop(columns=['episode']) 
    df = df_original[["volume", "vwap", "open", "close", "high", "low", "transactions", "tp_0.004"]].rename(columns={"tp_0.004": "label"}) # 0.01 0.001
    # df


    ####################################### Run Framework ##############################################################################################

    if data_form == 'frac_diff':
        X = df_fd.drop(columns=['label'])
        y = df_fd['label'] 
    elif data_form == 'original':
        X = df.drop(columns=['label'])
        y = df['label']
    elif data_form == 'first_order_diff':
        X = df.drop(columns=['label'])
        X[['open', 'high', 'low', 'close']] = X[['open', 'high', 'low', 'close']].diff()
        X.dropna(inplace=True)
        y = df['label'][1:]
    elif data_form == 'ta_original':
        X = df_ta.drop(columns=['label'])
        y = df_ta['label']
    elif data_form == 'ta_fod':
        X = df_ta.drop(['volume', 'transactions', 'label'], axis=1).diff()
        X = X.join(df_ta[['volume', 'transactions']]).dropna()
        y = df_ta['label'][1:]
    dataset_name = 'sp500'
    model_name = 'random_forest'
    chunk_size = 10_000
    # cold_start_size = 10_000
    num_runs = 2

    print(f'Running measurements with params: format={data_form},chunk_size={chunk_size}, num_runs={num_runs}, \
        dataset_name={dataset_name}, model_name={model_name}')

    a,c,p = run_measurements(X, y, chunk_size, dataset_name, model_name, num_runs=num_runs, frac_diff=False)


    ####################################### Save Results and Visualizations ###############################################################################
    import os
    base_dir = '/mnt/c/Users/resha/Documents/Github/balancing_framework/results/'
    subfolder = f"chunk_size={chunk_size} num_runs={num_runs} {dataset_name} {model_name}"
    save_dir = os.path.join(base_dir, subfolder)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(f'{save_dir}/adaptation_results_{data_form}.pkl', 'wb') as f:
        pickle.dump(a, f)
    with open(f'{save_dir}/consolidation_results_{data_form}.pkl', 'wb') as f:
        pickle.dump(c, f)

    end = time.time()
    print(f"Runtime: {(end - start) / 60} minutes")

    viz(a, c, metric='accuracy', title=data_form, dir=save_dir) 
    viz(a, c, metric='f1', title=data_form, dir=save_dir) 

