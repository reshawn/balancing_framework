import random
import time
import pandas as pd
from rocket_functions import generate_kernels, apply_kernels
from fracdiff import frac_diff_bestd
from evaluator import Evaluator

def my_train_test_split(X,y,test_size=0.2, ignore_size=0.25, random_state=777):
    # a train test split that does not randomly pull from the first 25% of data
    # because frac diff will drop values and we want the same test sets for with and without frac diff
    random.seed(random_state)  # Set the random seed
    n = round(len(X)*test_size)  # Number of random numbers
    start = round(len(X)*ignore_size)  # Start of range
    end = len(X)-1  # End of range

    random_numbers = random.sample(range(start, end + 1), n)
    all_numbers = set(range(start, end + 1))
    remaining_numbers = list(all_numbers - set(random_numbers))

    X_train, X_test = X.iloc[remaining_numbers], X.iloc[random_numbers]
    y_train, y_test = y.iloc[remaining_numbers], y.iloc[random_numbers]

    return X_train, X_test, y_train, y_test

def train_test_split_by_indices(X,y,test_indices, num_dropped=0):
    test_indices = test_indices - num_dropped
    # print(f'test_indices: {test_indices.min()} - {test_indices.max()}')
    # print(f'X: {X.index.min()} - {X.index.max()}')
    # print(f'y: {y.index.min()} - {y.index.max()}')
    X_test = X.loc[test_indices]
    y_test = y.loc[test_indices]
    X_train = X.drop(test_indices)
    y_train = y.drop(test_indices)
    return X_train, X_test, y_train, y_test

def run_measurements_episodic(datasets, titles, dataset_name, model_name, num_runs=10, test_size=None, frac_diff=False, rocket=False):

    # PREPROCESSING -------------------------------------------------------------------------------------------------

    # train test split for each episode

    episodes = []
    for X,y in datasets:
        X_train, X_test, y_train, y_test = my_train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        episode = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'X': X, 'y': y}
        episodes.append(episode)

    # Frac Diff
    if frac_diff:
        new_episodes = []
        for i, episode in enumerate(episodes):
            start = time.perf_counter()
            X, fd_change_pct = frac_diff_bestd(episode['X'])
            end = time.perf_counter()
            X.dropna(inplace=True)
            y = episode['y'].iloc[:len(X)]
            drop_pct = 1 - len(X) / len(episode['X'])
            time_taken_mins = (end-start)/60
            # ensure the test set is the same as without frac diff
            X_train, X_test, y_train, y_test = train_test_split_by_indices(X, y, episode['y_test'].index, num_dropped=0) # len(episode['X'])-len(X)
            # pack new episode, along with some frac diff info
            new_episode = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'X': X, 'y': y, 'fd_change_pct': fd_change_pct, 'drop_pct': drop_pct, 'fd_time_taken_mins': time_taken_mins}
            new_episodes.append(new_episode)
        episodes = new_episodes

    # Rocket
    if rocket:
        for i, episode in enumerate(episodes):
                input_length = episode['X_train'].shape[-1]
                kernels = generate_kernels(input_length, 10_000)
                start = time.perf_counter()
                X_train = apply_kernels(episode['X_train'].to_numpy(), kernels)
                X_test = apply_kernels(episode['X_test'].to_numpy(), kernels)
                end = time.perf_counter()

                time_taken_mins = (end-start)/60
                episode['X_train'] = pd.DataFrame(X_train)
                episode['X_test'] = pd.DataFrame(X_test)
                episode['rocket_time_taken_mins'] = time_taken_mins
    
    # Pack the preprocessing info
    prep_info = {}
    if frac_diff:
        prep_info['frac_diff'] = True
        prep_info['fd_change_pct'] = [ep['fd_change_pct'] for ep in episodes]
        prep_info['drop_pct'] = [ep['drop_pct'] for ep in episodes]
        prep_info['fd_time_taken_mins'] = [ep['fd_time_taken_mins'] for ep in episodes]
    if rocket:
        prep_info['rocket'] = True
        prep_info['rocket_time_taken_mins'] = [ep['rocket_time_taken_mins'] for ep in episodes]

    # EVALUATION -------------------------------------------------------------------------------------------------

    eval = Evaluator(dataset_name, model_name, num_runs, test_size)

    # ADAPTATION EPISODIC MEASURE LOOP 

    adaptation_results = pd.DataFrame()
    for i in range(1, len(episodes)):
        trained_on1 = episodes[:i+1]
        trained_on2 = episodes[:i]
        test = episodes[i]
        trained_on1_titles = titles[:i+1]
        trained_on2_titles = titles[:i]
        test_title = titles[i]

        result = eval.adaptation_measure_episodic(
            [ep['X_train'] for ep in trained_on1], 
            [ep['y_train'] for ep in trained_on1], 
            [ep['X_train'] for ep in trained_on2],
            [ep['y_train'] for ep in trained_on2],
            test['X_test'], test['y_test'], trained_on1_titles, trained_on2_titles, test_title
            )
        adaptation_results = pd.concat([adaptation_results, result], ignore_index=True)
    adaptation_results["dataset_name"] = eval.dataset_name


    # CONSOLIDATION MEASURE LOOP 

    consolidation_results = pd.DataFrame()
    consolidation_results_full = pd.DataFrame()
    for i in range(1, len(episodes)):
        trained_on1 = episodes[:i+1]
        trained_on2 = episodes[:i]
        test = episodes[:i]
        trained_on1_titles = titles[:i+1]
        trained_on2_titles = titles[:i]
        test_titles = titles[:i]

        result_avg, result_full = eval.consolidation_measure(
            [ep['X_train'] for ep in trained_on1], 
            [ep['y_train'] for ep in trained_on1], 
            [ep['X_train'] for ep in trained_on2],
            [ep['y_train'] for ep in trained_on2],
            [ep['X_test'] for ep in test], [ep['y_test'] for ep in test], trained_on1_titles, trained_on2_titles, test_titles
            )
        consolidation_results = pd.concat([consolidation_results, result_avg], ignore_index=True)
        consolidation_results_full = pd.concat([consolidation_results_full, result_full], ignore_index=True)
    consolidation_results["dataset_name"] = eval.dataset_name
    consolidation_results_full["dataset_name"] = eval.dataset_name


    return adaptation_results, consolidation_results, consolidation_results_full, prep_info



def run_measurements(X, y, chunk_size, cold_start_size, dataset_name, model_name, num_runs=10, frac_diff=False, rocket=False):

    # PREPROCESSING -------------------------------------------------------------------------------------------------

    # Frac Diff
    if frac_diff:
        start = time.perf_counter()
        old_len = len(X)
        X, fd_change_pct = frac_diff_bestd(X)
        end = time.perf_counter()
        X.dropna(inplace=True)
        y = y.iloc[:len(X)]
        num_dropped = old_len - len(X)
        time_taken_mins = (end-start)/60

    # Rocket
    if rocket:
        input_length = X.shape[-1]
        kernels = generate_kernels(input_length, 10_000)
        start = time.perf_counter()
        X = apply_kernels(X.to_numpy(), kernels)
        end = time.perf_counter()
        time_taken_mins = (end-start)/60
    
    # Pack the preprocessing info
    prep_info = {}
    if frac_diff:
        prep_info['frac_diff'] = True
        prep_info['fd_change_pct'] = fd_change_pct
        prep_info['num_dropped'] = num_dropped
        prep_info['fd_time_taken_mins'] = time_taken_mins
    if rocket:
        prep_info['rocket'] = True
        prep_info['rocket_time_taken_mins'] = time_taken_mins

    # EVALUATION -------------------------------------------------------------------------------------------------

    eval = Evaluator(dataset_name, model_name, num_runs, None)

    # ADAPTATION MEASURE LOOP 

    # in the case of frac diff or any rolling window method, remember to subtract the dropped rows from the cold start size
    adaptation_results = eval.adaptation_measure(X, y, chunk_size, cold_start_size)


    # CONSOLIDATION MEASURE LOOP 
    consolidation_results = eval.consolidation_measure(X, y, chunk_size, cold_start_size)

    return adaptation_results, consolidation_results, prep_info