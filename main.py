from steps.ingest_data import ingest_data
from steps.clean_data import clean_data

def main():
    churn_data = ingest_data(data_path="data/Churn_Modelling.csv")
    X_train, X_test, y_train, y_test = clean_data(churn_data)

if __name__ == "main":
    main()