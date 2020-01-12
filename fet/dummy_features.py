import pandas as pd


def generate_dummy_features(df: pd.DataFrame, dtype='object'):
    return pd.get_dummies(df, columns=df.select_dtypes(include=dtype).columns)
