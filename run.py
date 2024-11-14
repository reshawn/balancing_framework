import pickle
import pandas as pd
import numpy as np
import json
import time

from framework import run_measurements, viz


# Don't forget to point the output to a log file

start = time.time()

####################################### Load Data ##############################################################################################

with open('/mnt/c/Users/Joseph/Documents/Github/balancing_framework/spy5m_labelled_episodes.pkl', 'rb') as f:
    df_original = pickle.load(f)
with open('/mnt/c/Users/Joseph/Documents/Github/balancing_framework/spy5m_labelled_episodes_fracdiff.pkl', 'rb') as f:
    df_fd = pickle.load(f)
# PZ algorithm has some look ahead so remove the episode labels, will be uesd only for some kind of analysis afterwards
df = df_original.drop(columns=['episode']) 
# df


####################################### Run Framework ##############################################################################################

data_form = 'Original' # 'Frac_Diff' , 'First_Order_Diff', 'Original'

if data_form == 'Frac_Diff':
    X = df_fd.drop(columns=['label'])
    y = df_fd['label'] 
elif data_form == 'Original':
    X = df.drop(columns=['label'])
    y = df['label']
elif data_form == 'First_Order_Diff':
    X = df.drop(columns=['label'])
    X[['open', 'high', 'low', 'close']] = X[['open', 'high', 'low', 'close']].diff()
    y = df['label'][1:]
else:
    raise ValueError('Invalid data form setting')
dataset_name = 'sp500'
model_name = 'random_forest'
chunk_size = 50_000
cold_start_size = 10_000
num_runs = 10

print(f'Running measurements with params: format={data_form},chunk_size={chunk_size}, cold_start_size={cold_start_size}, num_runs={num_runs}, \
       dataset_name={dataset_name}, model_name={model_name}')

a,c,p = run_measurements(X, y, chunk_size, cold_start_size, dataset_name, model_name, num_runs=num_runs, frac_diff=False)


####################################### Save Results and Visualizations ###############################################################################

with open(f'/mnt/c/Users/Joseph/Documents/Github/balancing_framework/results/adaptation_results_{data_form}.pkl', 'wb') as f:
    pickle.dump(a, f)
with open(f'/mnt/c/Users/Joseph/Documents/Github/balancing_framework/results/consolidation_results_{data_form}.pkl', 'wb') as f:
    pickle.dump(c, f)

end = time.time()
print(f"Runtime: {(end - start) / 60} minutes")

viz(a, c, metric='accuracy', title=data_form) # Frac Diff , First Order Diff
viz(a, c, metric='f1', title=data_form) # Frac Diff , First Order Diff

