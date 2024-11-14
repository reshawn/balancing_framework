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

from sklearn.mixture import GaussianMixture

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

    best_params = tune(model_name, X_train, y_train, X_val, y_val, num_runs=num_runs)
    print(f'best_params: {best_params}')

    result = train_eval(model_name, best_params, X_train, y_train, X_test, y_test, num_runs=num_runs)

    result["mode"] = mode
    result["dataset_name"] = dataset_name
    # add result to results dataframe
    results = pd.concat([results,pd.DataFrame(result, index=[0])], ignore_index=True)
    
    return best_params, results


def tune(model_name, X_train, y_train, X_val, y_val, num_runs=10):
    # tune hyper params
    def objective(trial):
        sklearn_models = ['ridge_classifier', 'random_forest', 'logistic_regression']
        pytorch_models = []
        if model_name in sklearn_models:
            params = load_model_params(model_name, trial)
            clf = load_model(model_name, params)
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

def train_eval(model_name, best_params, X_train, y_train, X_test, y_test, num_runs=10):
    
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
        f1 = f1_score(y_test, clf.predict(X_test)) # , average='weighted' ; consider if setup changes to the more realistic multi-class imbalanced labels
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



import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def gen_chunk_base(df: pd.DataFrame, sample_size: int):
    """
    Generate a sample of data from a given dataframe using a Gaussian Mixture Model.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to generate a sample from.
    sample_size : int
        The number of samples to generate.

    Returns
    -------
    X : pd.DataFrame
        The generated sample without the label.
    y : pd.Series
        The generated sample's label.
    """
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df)
    # Split data into training and validation sets
    X_train, X_val = train_test_split(X, test_size=0.3, random_state=RANDOM_STATE)

    # Define the range of components to test
    n_components_range = range(1, 11)

    # Lists to store BIC and AIC scores
    bic_scores = []
    aic_scores = []
    gmms = []

    # Fit GMM models for each number of components and calculate BIC and AIC
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, random_state=RANDOM_STATE)
        gmm.fit(X_train)
        gmms.append(gmm)
        bic_scores.append(gmm.bic(X_val))
        aic_scores.append(gmm.aic(X_val))

    # Select the model with the lowest BIC
    optimal_components = n_components_range[np.argmin(bic_scores)]
    print(f"Optimal number of components according to BIC: {optimal_components}")

    gmm = gmms[optimal_components - 1]
    sample = gmm.sample(sample_size)[0]
    sample = scaler.inverse_transform(sample)
    sample = pd.DataFrame(sample, columns=df.columns)
    y = sample['label']
    # enforce a few simple logical rules on the generated sample
    y = y.round().abs()
    X = sample.drop('label', axis=1)
    X['high'] = X[['open', 'close', 'high', 'low']].max(axis=1)
    X['low'] = X[['open', 'close', 'high', 'low']].min(axis=1)

    return X,y

def gen_chunk(df: pd.DataFrame, sample_size: int, split_by_label: bool = False):
    # While inheriting labels by splitting the df by class and using separate GMMs:
    """
    Generate a sample chunk from the given DataFrame using Gaussian Mixture Models (GMMs).
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    sample_size (int): The desired size of the sample to generate.
    split_by_label (bool): If True, the DataFrame is split by class label and separate 
                           GMMs are used for each class to generate samples; otherwise,
                           a single GMM is used for the entire DataFrame.
    
    Returns:
    tuple: A tuple containing the generated features (X) and labels (y) as pandas DataFrames.
    """
    if split_by_label:
        sample_X_slices = []
        sample_y_slices = []
        for label in df.label.unique():
            df2 = df[df.label == label]
            sample_X_slice, _ = gen_chunk_base(df2, int(sample_size/len(df.label.unique())))
            sample_X_slices.append(sample_X_slice)
            sample_y_slices.append(pd.Series([label] * len(sample_X_slice)) ) # fill with repeated label to inherit from the class split
        X = pd.concat(sample_X_slices, axis=0)
        y = pd.concat(sample_y_slices, axis=0)
    # While Attempting to generate labels:
    else:
        X,y = gen_chunk_base(df, sample_size)
    
    return X,y
