import pandas as pd


def generate_dummy_features_by_columns(df: pd.DataFrame, columns: [str]):
    if columns:
        return pd.get_dummies(df, columns=columns)
    return df


def generate_dummy_features_by_data_type(df: pd.DataFrame, dtype: str):
    columns = list(df.select_dtypes(include=dtype).columns)
    return generate_dummy_features_by_columns(df, columns)


def generate_dummy_features_by_column_prefix(df: pd.DataFrame, prefix: str):
    columns = [col for col in df.columns if col.startswith(prefix)]
    return generate_dummy_features_by_columns(df, columns)

