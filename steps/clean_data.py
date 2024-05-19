import polars as pl
from sklearn.model_selection import train_test_split


def clean_data(df: pl.DataFrame) -> tuple:
    """
    Clean the given DataFrame by removing unnecessary columns and rows.

    Args:
        df (pl.DataFrame): The DataFrame to clean.

    Returns:
        pl.DataFrame: The cleaned DataFrame.
    """
    # Remove unnecessary columns
    DROP_COLUMNS = ['RowNumber', 'CustomerId', 'Surname']

    # Remove duplicates
    df = df.unique()

    # Remove rows with missing values
    df = df.drop_nulls()

    # Remove unnecessary columns
    df = df.drop(columns=DROP_COLUMNS)

    # Trasform geography and gender to dummy variables
    df = pl.concat(
        [df.drop(columns=['Geography', 'Gender']),
        df.select(['Geography', 'Gender']).to_dummies(drop_first=True)],
        how='horizontal'
    )

    # Split the data into training and testing sets
    X = df.drop(columns=['Exited'])
    y = df['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test