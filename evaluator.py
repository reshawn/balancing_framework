import pandas as pd
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from balancing_framework.training import tune, train_eval
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
    
    def consolidation_measure(
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