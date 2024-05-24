from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import xgboost as xgb
import matplotlib.pyplot as plt

def get_calibrated_model(model, X_train, y_train):
    """This function calibrates the model using isotonic regression and returns the calibrated model.

    Args:
        model (_type_): non-calibrated model
        X_train (_type_): train data
        y_train (_type_): train targets
    """
    params = model.get_params()
    xgb_model_cal = xgb.XGBClassifier()
    xgb_model_cal.set_params(**params)

    # Calibrate the model using isotonic regression
    cal_model = CalibratedClassifierCV(xgb_model_cal, cv=3, method='isotonic')
    cal_model.fit(X_train.to_numpy(), y_train.to_numpy())

    return cal_model

def get_calibration_curve(model, X_test, y_test) -> None:
    """This function returns the calibration curve for the model.

    Args:
        model (_type_): calibrated model
        X_train (_type_): train data
        y_train (_type_): train targets
    """
    prob_pos = model.predict_proba(X_test)[:, 1]
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)

    fig, axs = plt.subplots(nrows=2, ncols=1,figsize=(6, 2*3.84))

    axs[0].plot(mean_predicted_value, fraction_of_positives, "s-", label="XGBoost")
    axs[0].plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    axs[0].set_ylabel("Fraction of positives")
    axs[0].set_xlabel("Mean Predicted Probability")
    axs[0].set_title('calibration curve')
    axs[0].legend()

    axs[1].hist(prob_pos, range=(0, 1), bins=10, density=True, lw=2, alpha = 0.3)
    axs[1].set_xlabel("Estimated probability")
    axs[1].set_ylabel("Count")
    axs[1].set_title('Probability Distribution')

    plt.tight_layout()
    plt.show()
