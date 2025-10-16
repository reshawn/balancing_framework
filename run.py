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
    parser.add_argument('--forms', type=str, nargs='+', help="Data forms, one or more of: ['frac_diff' , 'first_order_diff', 'original']")
    args = parser.parse_args()

    data_forms = [ f.lower() for f in args.forms ]

    valid_forms = ['frac_diff' , 'first_order_diff', 'original', 'ta_original', 'ta_fod', 'ta_frac_diff']
    for data_form in data_forms:
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
    with open('/mnt/c/Users/resha/Documents/Github/balancing_framework/spy5m_ta_fracdiff.pkl', 'rb') as f:
        df_fd_ta = pickle.load(f)
        df_fd_ta['label'] = df_original['tp_0.004'][df_ta.index]
    # PZ algorithm has some look ahead so remove the episode labels if those are there, will be uesd only for some kind of analysis afterwards
    # df = df_original.drop(columns=['episode']) 
    df = df_original[["volume", "vwap", "open", "close", "high", "low", "transactions", "tp_0.004"]].rename(columns={"tp_0.004": "label"}) # 0.01 0.001
    # df


    ####################################### Run Framework ##############################################################################################
    X = pd.DataFrame()
    y = df['label']


    if 'frac_diff' in data_forms:
        X = X.join(df_fd.drop(columns=['label']).add_suffix(f'_fd'), how='outer')
    del df_fd
    if 'original' in data_forms:
        X = X.join(df.drop(columns=['label']).add_suffix(f'_o'), how='outer')
    if 'first_order_diff' in data_forms:
        # if wanted to omit cols from diff
        # diff = df.drop(['volume', 'transactions', 'label'], axis=1).diff()
        # diff = diff.join(df[['volume', 'transactions']])
        diff = df.drop(['label'], axis=1).diff().add_suffix(f'_fod')
        X = X.join(diff, how='outer')
    del df
    if 'ta_original' in data_forms:
        X = X.join(df_ta.drop(columns=['label']).add_suffix(f'_tao'), how='outer')
    if 'ta_fod' in data_forms:
        diff = df_ta.drop(['label'], axis=1).diff().add_suffix(f'_tafod')
        X = X.join(diff, how='outer')
    del df_ta
    if 'ta_frac_diff' in data_forms:
        X = X.join(df_fd_ta.drop(columns=['label']).add_suffix(f'_tafd'), how='outer')
    del df_fd_ta
    
            
            
    X.dropna(inplace=True)
    # drop y with index not in X
    y = y[y.index.isin(X.index)]



    dataset_name = 'sp500'
    model_name = 'random_forest'
    chunk_size = 10_000
    # cold_start_size = 10_000
    num_runs = 10

    print(f'Running measurements with params: format={data_forms},chunk_size={chunk_size}, num_runs={num_runs}, \
        dataset_name={dataset_name}, model_name={model_name}')

    a,c,p = run_measurements(X, y, chunk_size, dataset_name, model_name, num_runs=num_runs, frac_diff=False)


    ####################################### Save Results and Visualizations ###############################################################################
    import os
    base_dir = '/mnt/c/Users/resha/Documents/Github/balancing_framework/results/'
    subfolder = f"chunk_size={chunk_size} num_runs={num_runs} {dataset_name} {model_name}"
    save_dir = os.path.join(base_dir, subfolder)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(f'{save_dir}/adaptation_results_{data_forms}.pkl', 'wb') as f:
        pickle.dump(a, f)
    with open(f'{save_dir}/consolidation_results_{data_forms}.pkl', 'wb') as f:
        pickle.dump(c, f)

    end = time.time()
    print(f"Runtime: {(end - start) / 60} minutes")

    viz(a, c, metric='accuracy', title=data_form, dir=save_dir) 
    viz(a, c, metric='f1', title=data_form, dir=save_dir) 

