# Feature Extraction Toolkit

[Examples of usage](#examples-of-usage) •
[License](https://github.com/duketemon/fet/blob/master/LICENSE)

## Installation
### Install from PyPI
```bash
pip install fet
```
### Install from source code
```bash
git clone https://github.com/duketemon/fet.git
cd fet
python setup.py install
```

## Examples of usage
### Generation of dummy features
#### `generate_dummy_features_by_columns`
```python3
from fet import generate_dummy_features_by_columns

df = pd.DataFrame(data={'source': ['web', 'android', 'ios', 'other']})
df = generate_dummy_features_by_columns(df, ['source'])
print(df.head()) 
>>    source_android  source_ios  source_other  source_web
>> 0               0           0             0           1
>> 1               1           0             0           0
>> 2               0           1             0           0
>> 3               0           0             1           0
```

#### `generate_dummy_features_by_data_type`
```python3
from fet import generate_dummy_features_by_data_type

df = pd.DataFrame(data={
    'id': [1, 2, 3, 4],
    'source': ['web', 'android', 'ios', 'other']
})
df = generate_dummy_features_by_data_type(df, 'object')
print(df.head())
>>    id  source_android  source_ios  source_other  source_web
>> 0   1               0           0             0           1
>> 1   2               1           0             0           0
>> 2   3               0           1             0           0
>> 3   4               0           0             1           0
```

#### `generate_dummy_features_by_data_type`
```python3
from fet import generate_dummy_features_by_column_prefix

df = pd.DataFrame(data={
    'id': [1, 2, 3, 4],
    'user-platform': ['web', 'android', 'ios', 'other'],
    'user-plan': ['free', 'premium', 'free', 'free'],
})
df = generate_dummy_features_by_column_prefix(df, 'user-')
print(df.head())
>>    id  user-platform_android  ...  user-plan_free  user-plan_premium
>> 0   1                      0  ...               1                  0
>> 1   2                      1  ...               0                  1
>> 2   3                      0  ...               1                  0
>> 3   4                      0  ...               1                  0
```

### Generation of datetime features
#### `generate_date_features`
```python3
from fet import generate_date_features

df = pd.DataFrame(data={
    'id': [1, 2, 3],
    'date': ['2019-06-17', '2019-06-21', '2019-07-07'],
})

df = generate_date_features(df, 'date')
print(df.head())
>>    id  date__year  ...  date__is_year_end  date__is_weekend
>> 0   1        2019  ...                  0                 0
>> 1   2        2019  ...                  0                 0
>> 2   3        2019  ...                  0                 1
```

#### `generate_time_features`
```python3
from fet import generate_time_features

df = pd.DataFrame(data={
    'id': [1, 2, 3],
    'date': ['2019-06-17 12:23', '2019-06-21 01:34', '2019-07-07 21:13'],
})
df = generate_time_features(df, 'date')
print(df.head())
>>    id  date__hour  date__minute date__time_category
>> 0   1          12            23           afternoon
>> 1   2           1            34               night
>> 2   3          21            13             evening
```

#### `generate_datetime_features`
```python3
from fet import generate_datetime_features

df = pd.DataFrame(data={
    'id': [1, 2, 3],
    'date': ['2019-06-17 12:23', '2019-06-21 01:34', '2019-07-07 21:13'],
})
df = generate_datetime_features(df, 'date')
print(df.head())
>>    id  date__year  date__month  ...  date__hour  date__minute  date__time_category
>> 0   1        2019            6  ...          12            23            afternoon
>> 1   2        2019            6  ...           1            34                night
>> 2   3        2019            7  ...          21            13              evening
```
