from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model, save_model
from steps.evaluation import evaluate_model

from calibration.calibration import get_calibration_curve, get_calibrated_model

import joblib

def main():
    churn_data = ingest_data(data_path="data/Churn_Modelling.csv")
    X_train, X_test, y_train, y_test = clean_data(churn_data)
    xgb_model = train_model(X_train, y_train)
    save_model(xgb_model, "saved_models/xgb_model.joblib")
    xgb_model = joblib.load("saved_models/xgb_model.joblib")
    accuracy = evaluate_model(xgb_model, X_test, y_test)
    get_calibration_curve(xgb_model, X_test, y_test)

    xgb_model_calibrated = get_calibrated_model(xgb_model, X_test, y_test)
    accuracy_calibrated = evaluate_model(xgb_model_calibrated, X_test, y_test)
    get_calibration_curve(xgb_model_calibrated, X_test, y_test)

if __name__ == "main":
    main()