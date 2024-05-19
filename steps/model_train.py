import polars as pl
import xgboost as xgb

def train_model(X_train: pl.DataFrame, y_train: pl.Series):
    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    xgb_model.fit(X_train, y_train)

    return xgb_model