import polars as pl
import xgboost as xgb

from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, X_test: pl.DataFrame, y_test: pl.Series):
    """
    Evaluates the performance of a trained XGBoost model on a test set.

    Args:
        model (xgboost.XGBClassifier): Trained XGBoost model.
        X_test (pl.DataFrame): Test set features.
        y_test (pl.Series): Test set labels.

    Returns:
        float: Accuracy score on the test set.
    """
    
    # Make predictions on test data
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)
    
    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    
    return accuracy