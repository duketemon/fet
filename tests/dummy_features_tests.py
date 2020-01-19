import pytest
import numpy as np
import pandas as pd

from fet import generate_dummy_features_by_columns
from fet import generate_dummy_features_by_data_type
from fet import generate_dummy_features_by_column_prefix


# TESTS FOR generate_dummy_features_by_columns
def generate_dummy_features_by_columns__one_column_test():
    df = pd.DataFrame(data={'platform': ['web', 'android', 'ios', 'other']})
    df = generate_dummy_features_by_columns(df, ['platform'])
    assert np.array_equal(
        np.array(sorted(['platform_web', 'platform_android', 'platform_ios', 'platform_other'])),
        sorted(df.columns)
    )


def generate_dummy_features_by_columns__two_columns_test():
    df = pd.DataFrame(data={
        'cat-1': ['val-1', 'val-1', 'val-2'],
        'cat-2': ['val-1', 'val-2', 'val-2'],
        'cat-3': ['val-1', 'val-1', 'val-2'],
    })
    df = generate_dummy_features_by_columns(df, ['cat-1', 'cat-3'])
    assert np.array_equal(np.array(['cat-2', 'cat-1_val-1', 'cat-1_val-2', 'cat-3_val-1', 'cat-3_val-2']), df.columns)


def generate_dummy_features_by_columns__wrong_key_test():
    df = pd.DataFrame(data={'cat': ['val-1', 'val-1', 'val-2']})
    with pytest.raises(KeyError):
        generate_dummy_features_by_columns(df, ['catw'])


# TESTS FOR generate_dummy_features_by_data_type
def generate_dummy_features_by_data_type__one_column_test():
    df = pd.DataFrame(data={
        'id': [1, 2, 3, 4],
        'platform': ['web', 'android', 'ios', 'other']
    })
    df = generate_dummy_features_by_data_type(df, 'object')
    assert np.array_equal(
        np.array(sorted(['id', 'platform_web', 'platform_android', 'platform_ios', 'platform_other'])),
        sorted(df.columns)
    )


def generate_dummy_features_by_data_type__two_object_columns_test():
    df = pd.DataFrame(data={
        'cat-1': ['val-1', 'val-1', 'val-2'],
        'cat-2': [123123, 333333333, 111111],
        'cat-3': ['val-1', 'val-1', 'val-2'],
    })
    df = generate_dummy_features_by_data_type(df, 'object')
    assert np.array_equal(np.array(['cat-2', 'cat-1_val-1', 'cat-1_val-2', 'cat-3_val-1', 'cat-3_val-2']), df.columns)


# TESTS FOR generate_dummy_features_by_column_prefix
def generate_dummy_features_by_column_prefix__one_column_test():
    df = pd.DataFrame(data={
        'id': [1, 2, 3, 4],
        'user-platform': ['web', 'android', 'ios', 'other'],
        'user-plan': ['free', 'premium', 'free', 'free'],
    })
    df = generate_dummy_features_by_column_prefix(df, 'user-')
    print(df.head())
    assert np.array_equal(
        np.array(sorted(['id', 'user-plan_free', 'user-plan_premium',
                         'user-platform_web', 'user-platform_android', 'user-platform_ios', 'user-platform_other'])),
        sorted(df.columns)
    )


def generate_dummy_features_by_column_prefix__two_object_columns_test():
    df = pd.DataFrame(data={
        'id': [123123, 333333333, 111111],
        'cat-1': ['val-1', 'val-1', 'val-2'],
        'cat-3': ['val-1', 'val-1', 'val-2'],
    })
    df = generate_dummy_features_by_column_prefix(df, 'cat-')
    assert np.array_equal(np.array(['id', 'cat-1_val-1', 'cat-1_val-2', 'cat-3_val-1', 'cat-3_val-2']), df.columns)


if __name__ == '__main__':
    generate_dummy_features_by_columns__one_column_test()
    generate_dummy_features_by_columns__two_columns_test()
    generate_dummy_features_by_columns__wrong_key_test()

    generate_dummy_features_by_data_type__one_column_test()
    generate_dummy_features_by_data_type__two_object_columns_test()

    generate_dummy_features_by_column_prefix__one_column_test()
    generate_dummy_features_by_column_prefix__two_object_columns_test()
