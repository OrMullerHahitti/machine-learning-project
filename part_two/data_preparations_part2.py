from typing import Literal, List, Any, Iterable

import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.model_selection import train_test_split

# Read the CSV file
data = pd.read_csv('../prepared_data/cleaned_data_new.csv')

data = data.drop('SPEEDING_VIOLATIONS_Copy', axis=1)

print(data.head())

def cast_dataframe_to_int(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col not in ['Credit Score', 'Normalized Age and Experience Mean']:
            df[col] = df[col].astype('int')
    return df

def get_dummies(df,unordered_cols:Iterable[str]=('Family Status','Postal Code')):
    df_c=df.copy()
    encoded_data = pd.get_dummies(df_c, columns=unordered_cols)
    df_c.drop(labels=unordered_cols, axis=1, inplace=True)
    df_c = pd.concat([df_c, encoded_data], axis=1)
    return df_c

def prepare_data(df:DataFrame = data,dummies:bool=False) -> tuple:
    '''
    This function prepares the data for the clustering algorithm
    :param unordered_cols:
    :param df:
    :return: X_train -> y_train -> X_test -> y_test
    '''
# Split the data into train and test sets
    df= cast_dataframe_to_int(df)
    train_data, test_data = train_test_split(df, test_size=0.1, stratify=data['Claims Next Year'], random_state=42)
    if dummies:
        train_data=get_dummies(train_data)
        test_data=get_dummies(test_data)
    X_train = train_data.drop('Claims Next Year', axis=1)
    y_train = train_data['Claims Next Year']
    X_test = test_data.drop('Claims Next Year', axis=1)
    y_test = test_data['Claims Next Year']

    return X_train, y_train, X_test, y_test





df_check=prepare_data()
print(df_check[0].head())