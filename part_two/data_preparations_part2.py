from typing import Literal, List, Any, Iterable, Final

import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


#Categorial ordered/continuous cols not including the Binary variables
categorical_ordered_continuous_cols:Final[Iterable[str]] = ['Credit Score', 'Normalized Age and Experience Mean', 'Annual Mileage',
                                       'Past Accidents', 'Education', 'Income Category', 'Speeding Violations']
# Categorical unordered columns for dummy encoding
categorical_unordered_cols:Final[Iterable[str]] = ['Postal Code', 'Family Status']

# Read the CSV file
data = pd.read_csv('../prepared_data/cleaned_data_new.csv')

data = data.drop('SPEEDING_VIOLATIONS_Copy', axis=1)

print(data.head())

def cast_dataframe_to_int(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col not in ['Credit Score', 'Normalized Age and Experience Mean']:
            df[col] = df[col].astype('int')
    return df

def get_dummies(df,unordered_cols=categorical_unordered_cols):
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
    y_train = train_data['Claims Next Year']

    y_test = test_data['Claims Next Year']
    train_data = train_data.drop('Claims Next Year', axis=1)

    test_data = test_data.drop('Claims Next Year', axis=1)
    if dummies:
        train_data=get_dummies(train_data)
        test_data=get_dummies(test_data)

    return train_data, y_train, test_data, y_test


def normalize_dataset(dataset, columns, method='minmax'):
    """
    Normalize specified columns in a dataset using Min-Max Scaling or Standardization.

    Parameters:
        dataset (pd.DataFrame): Input pandas DataFrame.
        columns (list): List of column indices (or names) to normalize.
        method (str): Normalization method ('minmax' or 'standard'). Default is 'minmax'.

    Returns:
        pd.DataFrame: A new DataFrame with normalized values in the specified columns.
    """
    # Select normalization method
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid method. Choose 'minmax' or 'standard'.")

    # Apply normalization to specified columns
    if all(isinstance(col, int) for col in columns):
        column_names = dataset.columns[columns]
    else:
        column_names = columns

    # Create a copy of the dataset to avoid modifying the original
    normalized_data = dataset.copy()
    normalized_data[column_names] = scaler.fit_transform(normalized_data[column_names])

    return normalized_data

if __name__ == '__main__':
    df_check=prepare_data(dummies=True)
    print(df_check[0].head())
