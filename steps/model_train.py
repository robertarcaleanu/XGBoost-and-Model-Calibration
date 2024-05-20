import polars as pl
import xgboost as xgb
import joblib

from sklearn.model_selection import GridSearchCV 

def train_model(X_train: pl.DataFrame, y_train: pl.Series):
    """This function trains the XGBoost model

    Args:
        X_train (pl.DataFrame): training features
        y_train (pl.Series): training target values

    Returns:
        _type_: best model
    """
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.1, 0.3, 0.5],
        "n_estimators": [100, 200, 300],
        "min_split_loss": [0.0, 0.1, 0.2],
        "subsample": [0.5, 0.7, 1],
    }
    
    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    grid_search = GridSearchCV(xgb_model, param_grid, cv=3, verbose=3, scoring="accuracy")
    grid_search.fit(X_train, y_train)
    xgb_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return xgb_model

def save_model(model, model_path: str) -> None:
    """This function saves the model to a file

    Args:
        model (xgb.XGBClassifier): trained model
        model_path (str): path to save the model
    """
    joblib.dump(model, model_path)