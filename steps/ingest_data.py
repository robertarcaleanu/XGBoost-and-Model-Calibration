import polars as pl

def ingest_data(data_path: str) -> pl.DataFrame:
    """This function ingests the data from the given path

    Args:
        data_path (str): path to the data

    Returns:
        pl.DataFrame: loaded dataframe
    """

    df = pl.read_csv(data_path)

    return df