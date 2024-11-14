
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
RANDOM_STATE = 777


def load_model(model_name, params):
    '''
    Load model given text name
    '''
    if model_name == "ridge_classifier":
        clf = RidgeClassifier(alpha=params['alpha'], random_state=RANDOM_STATE)
    elif model_name == 'random_forest':
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=RANDOM_STATE)
    elif model_name == 'logistic_regression':
        clf = LogisticRegression(random_state=RANDOM_STATE)
    return clf

def load_model_params(model_name, trial):
    '''
    Load model hyperparameters given text name
    '''
    params = {}
    if model_name == "ridge_classifier":
        params['alpha'] = trial.suggest_float('alpha', 0.1, 10)
    elif model_name == 'random_forest':
        params['n_estimators'] = trial.suggest_int('n_estimators', 1, 50)
        params['max_depth'] = trial.suggest_int('max_depth', 2, 32)
    elif model_name == 'logistic_regression':
        pass
    
    params['RANDOM_STATE'] = RANDOM_STATE
    return params




