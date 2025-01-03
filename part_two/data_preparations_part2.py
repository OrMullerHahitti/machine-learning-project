import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.model_selection import train_test_split

# Read the CSV file
data = pd.read_csv('../prepared_data/cleaned_data.csv')

print(data.head())


def prepare_data(df:DataFrame = data) -> tuple:
# Split the data into train and test sets
    train_data, test_data = train_test_split(df, test_size=0.1, stratify=data['CLAIMS_INSURANCE_NEXT_YEAR'], random_state=42)



    X_train = train_data.drop('CLAIMS_INSURANCE_NEXT_YEAR', axis=1)
    y_train = train_data['CLAIMS_INSURANCE_NEXT_YEAR']
    X_test = test_data.drop('CLAIMS_INSURANCE_NEXT_YEAR', axis=1)
    y_test = test_data['CLAIMS_INSURANCE_NEXT_YEAR']

    return X_train, y_train, X_test, y_test

