import pytest
import numpy as np
import pandas as pd


from fet import generate_datetime_features
from fet import generate_date_features
from fet import generate_time_features


# TESTS FOR generate_date_features
def generate_date_features__drop_target_column_test():
    df = pd.DataFrame(data={
        'id': [1, 2, 3],
        'dt-1': ['2019-06-17', '2019-06-21', '2019-07-07'],
    })
    df = generate_date_features(df, 'dt-1', drop_target_column=False)
    assert 'dt-1' in df.columns
    cols = df.shape[1]
    df = generate_date_features(df, 'dt-1', drop_target_column=True)
    assert 'dt-1' not in df.columns
    assert cols - df.shape[1] == 1


def generate_date_features__nonexistent_column_test():
    df = pd.DataFrame(data={
        'id': [1, 2, 3],
        'date': ['2019-06-17', '2019-06-21', '2019-07-07'],
    })

    with pytest.raises(ValueError):
        generate_date_features(df, 'my-date')


# TESTS FOR generate_time_features
def generate_time_features__drop_target_column_test():
    df = pd.DataFrame(data={
        'id': [1, 2, 3],
        'dt-1': ['2019-06-17 12:23', '2019-06-21 01:34', '2019-07-07 21:13'],
    })
    df = generate_time_features(df, 'dt-1', drop_target_column=False)
    assert 'dt-1' in df.columns
    cols = df.shape[1]
    df = generate_time_features(df, 'dt-1', drop_target_column=True)
    assert 'dt-1' not in df.columns
    assert cols - df.shape[1] == 1


def generate_time_features__nonexistent_column_test():
    df = pd.DataFrame(data={
        'id': [1, 2, 3],
        'date': ['2019-06-17 12:23', '2019-06-21 01:34', '2019-07-07 21:13'],
    })
    with pytest.raises(ValueError):
        generate_time_features(df, 'my-date')


def generate_time_features__time_category_test():
    df = pd.DataFrame(data={
        'id': [1, 2, 3],
        'date': ['2019-06-17 12:23', '2019-06-21 01:34', '2019-07-07 21:13'],
    })
    df = generate_time_features(df, 'date')
    assert np.array_equal(np.array(['afternoon', 'night', 'evening']), df['date__time_category'].values)


# TESTS FOR generate_datetime_features
def generate_datetime_features__drop_target_column_test():
    df = pd.DataFrame(data={
        'id': [1, 2, 3],
        'dt-1': ['2019-06-17 12:23', '2019-06-21 01:34', '2019-07-07 21:13'],
    })
    df = generate_datetime_features(df, 'dt-1', drop_target_column=False)
    assert 'dt-1' in df.columns
    cols = df.shape[1]
    df = generate_datetime_features(df, 'dt-1', drop_target_column=True)
    assert 'dt-1' not in df.columns
    assert cols - df.shape[1] == 1


def generate_datetime_features__nonexistent_column_test():
    df = pd.DataFrame(data={
        'id': [1, 2, 3],
        'date': ['2019-06-17 12:23', '2019-06-21 01:34', '2019-07-07 21:13'],
    })
    with pytest.raises(ValueError):
        generate_datetime_features(df, 'my-date')


if __name__ == '__main__':
    generate_date_features__drop_target_column_test()
    generate_date_features__nonexistent_column_test()

    generate_time_features__drop_target_column_test()
    generate_time_features__nonexistent_column_test()
    generate_time_features__time_category_test()

    generate_datetime_features__drop_target_column_test()
    generate_datetime_features__nonexistent_column_test()
