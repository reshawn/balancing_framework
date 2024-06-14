import numpy as np
import pandas as pd
import time
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import optuna
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
RANDOM_STATE = 777
num_kernels = 10_000

from models import load_model, load_model_params


def run(X, y, dataset_name, model_name, num_runs=10, mode='normal', results=pd.DataFrame(), test_size=None):
    # Split the data
    # Because we are trying methods that drop rows, we need to make sure the test set is the same
    # Since its time series also better to preserve order
    if test_size is not None:
        X_test, y_test = (X[-test_size:], y[-test_size:])
        X, y = (X[:-test_size], y[:-test_size])
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    else:
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=RANDOM_STATE)

    best_params = tune(model_name, X_train, y_train, X_val, y_val)

    result = train_eval(model_name, best_params, X_train, y_train, X_test, y_test, num_runs=num_runs)

    result["mode"] = mode
    result["dataset_name"] = dataset_name
    # add result to results dataframe
    results = pd.concat([results,pd.DataFrame(result, index=[0])], ignore_index=True)
    
    return best_params, results


def tune(model_name, X_train, y_train, X_val, y_val, num_runs=10):
    # tune hyper params
    def objective(trial):
        params = load_model_params(model_name, trial)
        clf = load_model(model_name, params)
        sklearn_models = ['ridge_classifier', 'random_forest', 'logistic_regression']
        pytorch_models = []
        if model_name in sklearn_models:
            clf.fit(X_train, y_train)
        elif model_name in pytorch_models:
            pass
        else:
            raise ValueError(f"Model {model_name} not supported")
        return accuracy_score(y_val, clf.predict(X_val))

    time_a = time.perf_counter()
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=num_runs)
    time_b = time.perf_counter()
    # print the optimization time in minutes
    print(f"Optimization Time: {(time_b - time_a) / 60} minutes")

    # load best params
    best_params = study.best_params

    return best_params

def train_eval(model_name, best_params, X_train, y_train, X_test, y_test, num_runs=1):
    
    # train the model with the best params 10 times and store the mean and std accuracy, along with training and test times
    accs = []
    mccs = []
    f1s = []
    train_times = []
    test_times = []
    for i in range(num_runs):
        # -- training ----------------------------------------------------------
        time_a = time.perf_counter()
        clf = load_model(model_name, best_params)
        sklearn_models = ['ridge_classifier', 'random_forest', 'logistic_regression']
        pytorch_models = []
        if model_name in sklearn_models:
            clf.fit(X_train, y_train)
        elif model_name in pytorch_models:
            pass
        else:
            raise ValueError(f"Model {model_name} not supported")
        time_b = time.perf_counter()
        train_times.append(time_b - time_a)
        # -- test --------------------------------------------------------------
        time_a = time.perf_counter()
        acc = accuracy_score(y_test, clf.predict(X_test))
        mcc = matthews_corrcoef(y_test, clf.predict(X_test))
        f1 = f1_score(y_test, clf.predict(X_test), average='weighted')
        time_b = time.perf_counter()
        test_times.append(time_b - time_a)
        print(f"Run {i} Accuracy: {acc:.4f}")
        accs.append(acc)
        # mccs.append(mcc)
        f1s.append(f1)

    result = {
        "accuracy_mean": np.mean(accs),
        # "mcc_mean": np.mean(mccs),
        "f1_mean": np.mean(f1s),
        "accuracy_std": np.std(accs),
        # "mcc_std": np.std(mccs),
        "f1_std": np.std(f1s),
        "time_training_seconds": np.mean(train_times),
        "time_test_seconds": np.mean(test_times),
        "model_name": model_name,
    }
    return result