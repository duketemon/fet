import numpy as np
import pandas as pd


READABLE_DATE_PROPERTIES = {
    'year': 'year',
    'month': 'month',
    'week': 'week',
    'day': 'day',
    'dayofweek': 'day_of_week',
    'dayofyear': 'day_of_year',
    'quarter': 'quarter',
    'is_month_start': 'is_month_start',
    'is_month_end': 'is_month_end',
    'is_quarter_start': 'is_quarter_start',
    'is_quarter_end': 'is_quarter_end',
    'is_year_start': 'is_year_start',
    'is_year_end': 'is_year_end',
}

TIME_CATEGORIES_RANGES = {
    'morning': (6, 11),
    'afternoon': (11, 17),
    'evening': (17, 23),
}

READABLE_TIME_PROPERTIES = {
    'hour': 'hour',
    'minute': 'minute',
}

VALUE_SEPARATOR = '__'
WEEKEND_DAYS = ['Saturday', 'Sunday']


def generate_datetime_features(
        df: pd.DataFrame,
        target_column_name: str,
        separator=VALUE_SEPARATOR,
        drop_target_column=True,
        weekend_days=WEEKEND_DAYS
):

    generate_date_features(df, target_column_name, separator, False, weekend_days)
    generate_time_features(df, target_column_name, separator, False)

    if drop_target_column:
        df.drop(target_column_name, axis=1, inplace=True)
    return df


def generate_date_features(
        df: pd.DataFrame,
        target_column_name: str,
        separator=VALUE_SEPARATOR,
        drop_target_column=True,
        weekend_days=WEEKEND_DAYS
):
    if target_column_name not in df.columns:
        raise ValueError(f'There is no column with name "{target_column_name}" in the dataframe')

    if not np.issubdtype(df[target_column_name].dtype, np.datetime64):
        df[target_column_name] = pd.to_datetime(df[target_column_name], infer_datetime_format=True)

    READABLE_DATE_PROPERTIES['weekday_name'] = 'weekday_name'
    for prop, readable_prop in READABLE_DATE_PROPERTIES.items():
        feature_name = target_column_name + separator + readable_prop
        df[feature_name] = getattr(df[target_column_name].dt, prop)
        if df[feature_name].dtype == 'bool':
            df[feature_name] = df[feature_name].astype(int)
    if weekend_days is not None:
        feature_name = target_column_name + separator + READABLE_DATE_PROPERTIES['weekday_name']
        df[target_column_name + separator + 'is_weekend'] = np.where(df[feature_name].isin(weekend_days), 1, 0)
    df.drop(target_column_name + separator + READABLE_DATE_PROPERTIES['weekday_name'], axis=1, inplace=True)
    READABLE_DATE_PROPERTIES.pop('weekday_name')

    if drop_target_column:
        df.drop(target_column_name, axis=1, inplace=True)
    return df


def get_time_category(hour: int, time_categories_ranges: dict):
    for category, ranges in time_categories_ranges.items():
        if ranges[0] < hour <= ranges[1]:
            return category
    return 'night'


def generate_time_features(
        df: pd.DataFrame,
        target_column_name: str,
        separator=VALUE_SEPARATOR,
        drop_target_column=True,
        time_categories_ranges=TIME_CATEGORIES_RANGES
):
    if target_column_name not in df.columns:
        raise ValueError(f'There is no column with name "{target_column_name}" in the dataframe')

    if not np.issubdtype(df[target_column_name].dtype, np.datetime64):
        df[target_column_name] = pd.to_datetime(df[target_column_name], infer_datetime_format=True)

    for prop in READABLE_TIME_PROPERTIES:
        feature_name = target_column_name + separator + prop
        df[feature_name] = getattr(df[target_column_name].dt, prop.lower())
        if df[feature_name].dtype == 'bool':
            df[feature_name] = df[feature_name].astype(int)
    if time_categories_ranges is not None:
        df[target_column_name + separator + 'time_category'] = \
            df[target_column_name + separator + READABLE_TIME_PROPERTIES['hour']]\
                .apply(lambda h: get_time_category(h, time_categories_ranges))

    if drop_target_column:
        df.drop(target_column_name, axis=1, inplace=True)
    return df
