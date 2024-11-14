import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')



from sklearn.model_selection import train_test_split

from training import tune, train_eval, gen_chunk

RANDOM_STATE = 777
num_kernels = 10_000


class Evaluator:
    def __init__(self, dataset_name, model_name, num_runs=10, test_size=None):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.num_runs = num_runs
        self.test_size = test_size
        self.results = pd.DataFrame()
        self.best_params = None

    def data_split(self, X, y, test=True):
        if test:
            if self.test_size is not None:
                X_test, y_test = (X[-self.test_size:], y[-self.test_size:])
                X, y = (X[:-self.test_size], y[:-self.test_size])
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
            else:
                X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
                X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=RANDOM_STATE)
            return X_train, X_val, y_train, y_val, X_test, y_test
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
            return X_train, X_val, y_train, y_val
        
    def preprocessing(self):
        pass

    
    
    def adaptation_measure(
            self,
            X: pd.DataFrame,
            y: pd.DataFrame,
            chunk_size: int,
            cold_start_size: int,
            gen: bool = False
            ):

        """
        Given a dataset, X and its labels, y, simulate a simple online setup where new instances are processed for change in chunks of size chunk_size.
        
        We define "adaptation" as the improvement in performance on new data when that new data is added to the training.
        With a set cold_start size, we iterate through the remaining data in chunks, training on the seen data and testing on the new chunk.
        The seen data is split for a validation set to tune hyperparameters.
        The list of f1 scores for each chunk is returned.

        Parameters
        ----------
        X : pd.DataFrame
            The input data
        y : pd.DataFrame
            The labels of the data
        chunk_size : int
            The size of the chunks of data to process at once
        cold_start_size : int
            The number of instances to use for the initial training of the model
        gen : bool, optional
            Whether to generate a new chunk of data, by default False

        Returns
        -------
        f1_scores : list
            A list of the f1 scores of the model on the new data at each iteration
        """
        X_seen, y_seen = X[:cold_start_size], y[:cold_start_size]
        X_unseen, y_unseen = X[cold_start_size:], y[cold_start_size:]

        # iterate through the unseen data in chunks
        all_results = []
        for i in tqdm(range(0, len(X_unseen), chunk_size)):
            X_chunk, y_chunk = X_unseen[i:i+chunk_size], y_unseen[i:i+chunk_size]
            # the new chunk gets added to the seen for testing, keep aside 10% for testing the adaptation if gen is false
            # else call gen_chunk using the new chunk
            if gen:
                df = X_chunk.copy()
                df['label'] = y_chunk
                X_chunk_test, y_chunk_test = gen_chunk(df, sample_size=chunk_size)
            else:
                X_chunk, X_chunk_test, y_chunk, y_chunk_test = train_test_split(X_chunk, y_chunk, test_size=0.1, random_state=RANDOM_STATE)

            X_seen, y_seen = pd.concat([X_seen, X_chunk]), pd.concat([y_seen, y_chunk])
            X_train, X_val, y_train, y_val = self.data_split(X_seen, y_seen, test=False)
            print(f'Tuning run {i/chunk_size} of {len(X_unseen)/chunk_size}')
            best_params = tune(self.model_name, X_train, y_train, X_val, y_val, num_runs=self.num_runs)
            print(f'Training run {i/chunk_size} of {len(X_unseen)/chunk_size}')
            result = train_eval(self.model_name, best_params, X_train, y_train, X_chunk_test, y_chunk_test, num_runs=self.num_runs)
            result['last_ts'] = X_unseen[i:i+chunk_size].index[-1]
            all_results.append(result)
        
        return all_results
    

    def consolidation_measure(
            self,
            X: pd.DataFrame,
            y: pd.DataFrame,
            chunk_size: int,
            cold_start_size: int
            ):
        '''
        Given a dataset, X and its labels, y, simulate a simple online setup where new instances are processed for change in chunks of size chunk_size.
        (The same setup as the adaptation measure)
        We define "consolidation" as the improvement or maintainence in performance on old data when new data is added to the training.
        Using the data already seen, we generate a representative set, similar to a replay buffer, for testing.
        With a set cold_start size, we iterate through the remaining data in chunks, training on the seen data and testing on the buffer.
        The seen data is split for a validation set to tune hyperparameters.
        The list of f1 scores for each chunk is returned.
        '''

        X_seen, y_seen = X[:cold_start_size], y[:cold_start_size]
        X_unseen, y_unseen = X[cold_start_size:], y[cold_start_size:]

        # iterate through the unseen data in chunks
        all_results = []
        for i in tqdm(range(0, len(X_unseen), chunk_size)):
            X_chunk, y_chunk = X_unseen[i:i+chunk_size], y_unseen[i:i+chunk_size]
            X_seen, y_seen = pd.concat([X_seen, X_chunk]), pd.concat([y_seen, y_chunk])
            X_train, X_val, y_train, y_val = self.data_split(X_seen, y_seen, test=False)
            print(f'Tuning run {i/chunk_size} of {len(X_unseen)/chunk_size}')
            best_params = tune(self.model_name, X_train, y_train, X_val, y_val, num_runs=self.num_runs)

            # generate a representative set from the seen data
            print(f'Generating chunk for run {i/chunk_size} of {len(X_unseen)/chunk_size}')
            df = X_seen.copy()
            df['label'] = y_seen
            X_gen, y_gen = gen_chunk(df, sample_size=chunk_size)

            print(f'Training run {i/chunk_size} of {len(X_unseen)/chunk_size}')
            result = train_eval(self.model_name, best_params, X_train, y_train, X_gen, y_gen, num_runs=self.num_runs)
            result['last_ts'] = X_chunk.index[-1]
            all_results.append(result)
        
        return all_results


# NO LONGER BEING USED
    def adaptation_measure_episodic(
            self,
            X_trained_on1: list[pd.DataFrame], 
            y_trained_on1: list[pd.DataFrame], 
            X_trained_on2: list[pd.DataFrame],
            y_trained_on2: list[pd.DataFrame],
            X_test_df: pd.DataFrame,
            y_test_df: pd.DataFrame,
            trained_on1_titles: list[str],
            trained_on2_titles: list[str],
            test_titles: list[str]
            ):
        '''
        This version works with given lists of predefined episodes of data and their titles.
        X_trained_on1 and 2 are lists of the X_train splits (inc. valid) from the various episodes/subsets of data
        similarly for y_trained_on1 and 2
        X_test_df and y_test_df instead are the X_test and y_test splits from the new data episode/subset
        We define "adaptation" as the improvement in performance on new data when that new data is added to the training.
        For example, the difference in accuracy on B when trained on A and B vs. the accuracy on B when trained on A.
        i.e [A,B] on B_test vs. [A] on B_test. where X_trained_on1 is [A,B] and X_trained_on2 is [A] and X_test_df is B
        X_trained_on1 is expected to be inclusive of the new data's train set
        The difference in metrics between the two training sets is the adaptation measure returned.
        The diff is [A,B] on b - [A] on b for each metric, so the change in performance on b with adding B.
        '''

        # train on trained_on1
        X = pd.concat(X_trained_on1)
        y = pd.concat(y_trained_on1)
        X_train, X_val, y_train, y_val = self.data_split(X, y, test=False)
        best_params = tune(self.model_name, X_train, y_train, X_val, y_val, num_runs=self.num_runs)
        result1 = train_eval(self.model_name, best_params, X_train, y_train, X_test_df, y_test_df)

        # train on trained_on2
        X = pd.concat(X_trained_on2)
        y = pd.concat(y_trained_on2)
        X_train, X_val, y_train, y_val = self.data_split(X, y, test=False)
        best_params = tune(self.model_name, X_train, y_train, X_val, y_val, num_runs=self.num_runs)
        result2 = train_eval(self.model_name, best_params, X_train, y_train, X_test_df, y_test_df)

        # calculate adaptation measure
        adaptation = {}
        adaptation["title"] = f'Trained on ({trained_on1_titles}) vs. ({trained_on2_titles}) Tested on ({test_titles})'
        for key in result1.keys():
            try:
                adaptation[key] = result1[key] - result2[key]
            except Exception as e:
                if type(e) == TypeError: 
                    # most likely a string label and not a metric, these SHOULD be the same for both results, so can assign
                    adaptation[key] = result1[key]
        adaptation["trained_on1_acc"] = result1["accuracy_mean"]
        adaptation["trained_on2_acc"] = result2["accuracy_mean"]
        adaptation["train_on1_f1"] = result1["f1_mean"]
        adaptation["train_on2_f1"] = result2["f1_mean"]
        
        return pd.DataFrame(adaptation, index=[0])

    
# NO LONGER BEING USED
    def consolidation_measure_episodic(
            self,
            X_trained_on1: list[pd.DataFrame], 
            y_trained_on1: list[pd.DataFrame], 
            X_trained_on2: list[pd.DataFrame],
            y_trained_on2: list[pd.DataFrame],
            X_test: list[pd.DataFrame],
            y_test: list[pd.DataFrame],
            trained_on1_titles: list[str],
            trained_on2_titles: list[str],
            test_titles: list[str]
            ):
        '''
        X_trained_on1 and 2 are lists of the X_train splits (inc. valid) from the various episodes/subsets of data
        similarly for y_trained_on1 and 2
        X_test and y_test instead are the X_test and y_test splits from the old data episodes/subsets
        We define "consolidation" as the improvement or maintainence in performance on old data when new data is added to the training.
        For example, the difference in accuracy on A when trained on A and B vs. the accuracy on A when trained on A.
        i.e [A,B] on A_test vs. [A] on A_test. where X_trained_on1 is [A,B] and X_trained_on2 is [A] and X_test is [A_test]
        Unlike with adaptation, we do not assume a single episode for test here since the old data can be from multiple episodes.
        X_trained_on1 is expected to be inclusive of the new data's train set while X_trained_on2 is without it
        X_test and y_test are expected to be the test sets from the old data episodes, i.e. the test sets of X_trained_on2's source episodes
        The difference in metrics between the two training sets is the consolidation measure returned
        The diff is [A,B] on a - [A] on a for each metric, so the change in performance on a with adding B.
        Returns the average consolidation across all test sets as one df and the consolidation for each test set as another.
        '''
        # combine the test sets
        X_test_combined = pd.concat(X_test)
        y_test_combined = pd.concat(y_test)

        # train on trained_on1
        X = pd.concat(X_trained_on1)
        y = pd.concat(y_trained_on1)
        X_train, X_val, y_train, y_val = self.data_split(X, y, test=False)
        best_params = tune(self.model_name, X_train, y_train, X_val, y_val, num_runs=self.num_runs)
        result1 = train_eval(self.model_name, best_params, X_train, y_train, X_test_combined, y_test_combined)
        result1_sublist = []
        for (X_test_df, y_test_df) in zip(X_test, y_test):
            result = train_eval(self.model_name, best_params, X_train, y_train, X_test_df, y_test_df)
            result1_sublist.append(result)

        # train on trained_on2
        X = pd.concat(X_trained_on2)
        y = pd.concat(y_trained_on2)
        X_train, X_val, y_train, y_val = self.data_split(X, y, test=False)
        best_params = tune(self.model_name, X_train, y_train, X_val, y_val, num_runs=self.num_runs)
        result2 = train_eval(self.model_name, best_params, X_train, y_train, X_test_combined, y_test_combined)
        result2_sublist = []
        for (X_test_df, y_test_df) in zip(X_test, y_test):
            result = train_eval(self.model_name, best_params, X_train, y_train, X_test_df, y_test_df)
            result2_sublist.append(result)

        # calculate consolidation measures
        consolidation = []
        cons_combined = {}
        cons_combined["title"] = f'Trained on ({trained_on1_titles}) vs. ({trained_on2_titles}) Tested on ({test_titles})'
        for key in result1.keys():
            try:
                cons_combined[key] = result1[key] - result2[key]
            except Exception as e:
                if type(e) == TypeError: 
                    # most likely a string label and not a metric, these SHOULD be the same for both results, so can assign
                    cons_combined[key] = result1[key]
        cons_combined["trained_on1_acc"] = result1["accuracy_mean"]
        cons_combined["trained_on2_acc"] = result2["accuracy_mean"]
        cons_combined["train_on1_f1"] = result1["f1_mean"]
        cons_combined["train_on2_f1"] = result2["f1_mean"]

        for i, (result1, result2) in enumerate(zip(result1_sublist, result2_sublist)):
            cons = {}
            cons["title"] = f'Trained on ({trained_on1_titles}) vs. ({trained_on2_titles}) Tested on ({test_titles[i]})'
            for key in result1.keys():
                try:
                    cons[key] = result1[key] - result2[key]
                except Exception as e:
                    if type(e) == TypeError: 
                        # most likely a string label and not a metric, these SHOULD be the same for both results, so can assign
                        cons[key] = result1[key]
            consolidation.append(cons)

        return pd.DataFrame(cons_combined, index=[0]), pd.DataFrame(consolidation)