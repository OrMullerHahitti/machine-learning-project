import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from scipy.stats import spearmanr, pearsonr, chi2_contingency, pointbiserialr
from tkinter import Tk, filedialog


train=pd.read_csv("XY_train.csv", encoding='latin1')

# Renaming the column OUTCOME to CLAIMS_INSURANCE_NEXT_YEAR
train.rename(columns={'OUTCOME': 'CLAIMS_INSURANCE_NEXT_YEAR'}, inplace=True)


test=pd.read_csv("X_test.csv", encoding='latin1')

test.rename(columns={'OUTCOME': 'CLAIMS_INSURANCE_NEXT_YEAR'}, inplace=True)

######################################################################################################################
# Analyze the 'ID' variable in the train dataset
id_unique_count = train['ID'].nunique()  # Number of unique values
id_all_numeric = pd.to_numeric(train['ID'], errors='coerce').notnull().all()  # Check if all non-null values are numeric
id_missing_count = train['ID'].isnull().sum()  # Count of missing values
id_all_unique = id_unique_count == train['ID'].count()  # Check if all values are unique
id_all_integers = train['ID'].dropna().apply(lambda x: float(x).is_integer()).all()  # Check if all non-null values are integers

# Analyze the 'ID' variable in the test dataset
id_unique_count_test = test['ID'].nunique()  # Number of unique values
id_all_numeric_test = pd.to_numeric(test['ID'], errors='coerce').notnull().all()  # Check if all non-null values are numeric
id_missing_count_test = test['ID'].isnull().sum()  # Count of missing values
id_all_unique_test = id_unique_count_test == test['ID'].count()  # Check if all values are unique
id_all_integers_test = test['ID'].dropna().apply(lambda x: float(x).is_integer()).all()  # Check if all non-null values are integers

# Display results
print("\nTrain ID analysis:")
print("Unique ID values:", id_unique_count)
print("All values numeric (excluding NULL):", id_all_numeric)
print("Missing values in ID:", id_missing_count)
print("All values unique:", id_all_unique)
print("All values integers:", id_all_integers)


print("\nTest ID analysis:")
print("Unique ID values (test):", id_unique_count_test)
print("All values numeric (excluding NULL) (test):", id_all_numeric_test)
print("Missing values in ID (test):", id_missing_count_test)
print("All values unique (test):", id_all_unique_test)
print("All values integers (test):", id_all_integers_test)

######################################################################################################################

# Analyze the 'GENDER' variable in the train dataset
gender_unique_count_train = train['GENDER'].nunique()  # Count unique values
gender_has_female_male_train = all(val in ["female", "male"] for val in train['GENDER'].dropna().unique())  # Check if all values are in ["female", "male"]
gender_missing_count_train = train['GENDER'].isnull().sum()  # Count missing values
gender_unique_values_train = train['GENDER'].dropna().unique()  # Unique values in train dataset


# Analyze the 'GENDER' variable in the test dataset
gender_unique_count_test = test['GENDER'].nunique()  # Count unique values
gender_has_female_male_test = all(val in ["female", "male"] for val in test['GENDER'].dropna().unique())  # Check if all values are in ["female", "male"]
gender_missing_count_test = test['GENDER'].isnull().sum()  # Count missing values
gender_unique_values_test = test['GENDER'].dropna().unique()  # Unique values in test dataset

# Display results
print("\nTrain GENDER analysis:")
print("Unique values:", gender_unique_values_train)
print("Unique count:", gender_unique_count_train)
print("All values are 'female' or 'male':", gender_has_female_male_train)
print("Missing values:", gender_missing_count_train)

print("\nTest GENDER analysis:")
print("Unique values:", gender_unique_values_test)
print("Unique count:", gender_unique_count_test)
print("All values are 'female' or 'male':", gender_has_female_male_test)
print("Missing values:", gender_missing_count_test)

######################################################################################################################

# Analyze the 'EDUCATION' variable in the train dataset
education_unique_count_train = train['EDUCATION'].nunique()  # Count unique values
education_has_expected_values_train = all(val in ['high school', 'none', 'university'] for val in train['EDUCATION'].dropna().unique())  # Check if all values are in the expected list
education_missing_count_train = train['EDUCATION'].isnull().sum()  # Count missing values
education_unique_values_train = train['EDUCATION'].dropna().unique()  # Unique values in train dataset

# Analyze the 'EDUCATION' variable in the test dataset
education_unique_count_test = test['EDUCATION'].nunique()  # Count unique values
education_has_expected_values_test = all(val in ['high school', 'none', 'university'] for val in test['EDUCATION'].dropna().unique())  # Check if all values are in the expected list
education_missing_count_test = test['EDUCATION'].isnull().sum()  # Count missing values
education_unique_values_test = test['EDUCATION'].dropna().unique()  # Unique values in test dataset

# Display results
print("\nTrain EDUCATION analysis:")
print("Unique values:", education_unique_values_train)
print("Unique count:", education_unique_count_train)
print("All values are in ['high school', 'none', 'university']:", education_has_expected_values_train)
print("Missing values:", education_missing_count_train)

print("\nTest EDUCATION analysis:")
print("Unique values:", education_unique_values_test)
print("Unique count:", education_unique_count_test)
print("All values are in ['high school', 'none', 'university']:", education_has_expected_values_test)
print("Missing values:", education_missing_count_test)

######################################################################################################################

# Analyze the 'INCOME' variable in the train dataset
income_unique_count_train = train['INCOME'].nunique()  # Count unique values
income_has_expected_values_train = all(val in ['upper class', 'poverty', 'middle class', 'working class'] for val in train['INCOME'].dropna().unique())  # Check if all values are in the expected list
income_missing_count_train = train['INCOME'].isnull().sum()  # Count missing values
income_unique_values_train = train['INCOME'].dropna().unique()  # Unique values in train dataset

# Analyze the 'INCOME' variable in the test dataset
income_unique_count_test = test['INCOME'].nunique()  # Count unique values
income_has_expected_values_test = all(val in ['upper class', 'poverty', 'middle class', 'working class'] for val in test['INCOME'].dropna().unique())  # Check if all values are in the expected list
income_missing_count_test = test['INCOME'].isnull().sum()  # Count missing values
income_unique_values_test = test['INCOME'].dropna().unique()  # Unique values in test dataset

# Display results
print("\nTrain INCOME analysis:")
print("Unique values:", income_unique_values_train)
print("Unique count:", income_unique_count_train)
print("All values are in ['upper class', 'poverty', 'middle class', 'working class']:", income_has_expected_values_train)
print("Missing values:", income_missing_count_train)

print("\nTest INCOME analysis:")
print("Unique values:", income_unique_values_test)
print("Unique count:", income_unique_count_test)
print("All values are in ['upper class', 'poverty', 'middle class', 'working class']:", income_has_expected_values_test)
print("Missing values:", income_missing_count_test)

######################################################################################################################

# Analyze the 'CREDIT_SCORE' variable in the train dataset
credit_unique_count_train = train['CREDIT_SCORE'].nunique()  # Count unique values
credit_has_expected_values_train = train['CREDIT_SCORE'].dropna().apply(lambda x: 0 <= x <= 1).all()  # Check if all values are between 0 and 1
credit_missing_count_train = train['CREDIT_SCORE'].isnull().sum()  # Count missing values
credit_unique_values_train = train['CREDIT_SCORE'].dropna().unique()  # Unique values in train dataset

# Analyze the 'CREDIT_SCORE' variable in the test dataset
credit_unique_count_test = test['CREDIT_SCORE'].nunique()  # Count unique values
credit_has_expected_values_test = test['CREDIT_SCORE'].dropna().apply(lambda x: 0 <= x <= 1).all()  # Check if all values are between 0 and 1
credit_missing_count_test = test['CREDIT_SCORE'].isnull().sum()  # Count missing values
credit_unique_values_test = test['CREDIT_SCORE'].dropna().unique()  # Unique values in test dataset

# Display results
print("\nTrain CREDIT_SCORE analysis:")
print("Unique Count:", credit_unique_count_train)
print("Has Expected Values (0-1):", credit_has_expected_values_train)
print("Missing Count:", credit_missing_count_train)
print("Unique Values (first 10):", credit_unique_values_train[:10])

print("\nTest CREDIT_SCORE analysis:")
print("Unique Count:", credit_unique_count_test)
print("Has Expected Values (0-1):", credit_has_expected_values_test)
print("Missing Count:", credit_missing_count_test)
print("Unique Values (first 10):", credit_unique_values_test[:10])

######################################################################################################################

# Analyze the 'VEHICLE_OWNERSHIP' variable in the train dataset
ownership_unique_count_train = train['VEHICLE_OWNERSHIP'].nunique()  # Count unique values
ownership_has_expected_values_train = train['VEHICLE_OWNERSHIP'].dropna().isin([0, 1]).all()  # Check if all values are binary (0/1)
ownership_missing_count_train = train['VEHICLE_OWNERSHIP'].isnull().sum()  # Count missing values
ownership_unique_values_train = train['VEHICLE_OWNERSHIP'].dropna().unique()  # Unique values in train dataset

# Analyze the 'VEHICLE_OWNERSHIP' variable in the test dataset
ownership_unique_count_test = test['VEHICLE_OWNERSHIP'].nunique()  # Count unique values
ownership_has_expected_values_test = test['VEHICLE_OWNERSHIP'].dropna().isin([0, 1]).all()  # Check if all values are binary (0/1)
ownership_missing_count_test = test['VEHICLE_OWNERSHIP'].isnull().sum()  # Count missing values
ownership_unique_values_test = test['VEHICLE_OWNERSHIP'].dropna().unique()  # Unique values in test dataset

# Display results
print("\nTrain VEHICLE_OWNERSHIP analysis:")
print("Unique Count:", ownership_unique_count_train)
print("Has Expected Values (Binary 0/1):", ownership_has_expected_values_train)
print("Missing Count:", ownership_missing_count_train)
print("Unique Values:", ownership_unique_values_train)

print("\nTest VEHICLE_OWNERSHIP analysis:")
print("Unique Count:", ownership_unique_count_test)
print("Has Expected Values (Binary 0/1):", ownership_has_expected_values_test)
print("Missing Count:", ownership_missing_count_test)
print("Unique Values:", ownership_unique_values_test)

######################################################################################################################

# Analyze the 'VEHICLE_YEAR' variable in the train dataset
vehicle_year_unique_count_train = train['VEHICLE_YEAR'].nunique()  # Count unique values
vehicle_year_has_expected_values_train = train['VEHICLE_YEAR'].dropna().isin(["after 2015", "before 2015"]).all()  # Check if all values are in the expected list
vehicle_year_missing_count_train = train['VEHICLE_YEAR'].isnull().sum()  # Count missing values
vehicle_year_unique_values_train = train['VEHICLE_YEAR'].dropna().unique()  # Unique values in train dataset

# Analyze the 'VEHICLE_YEAR' variable in the test dataset
vehicle_year_unique_count_test = test['VEHICLE_YEAR'].nunique()  # Count unique values
vehicle_year_has_expected_values_test = test['VEHICLE_YEAR'].dropna().isin(["after 2015", "before 2015"]).all()  # Check if all values are in the expected list
vehicle_year_missing_count_test = test['VEHICLE_YEAR'].isnull().sum()  # Count missing values
vehicle_year_unique_values_test = test['VEHICLE_YEAR'].dropna().unique()  # Unique values in test dataset

# Display results
print("\nTrain VEHICLE_YEAR analysis:")
print("Unique Count:", vehicle_year_unique_count_train)
print("Has Expected Values (['after 2015', 'before 2015']):", vehicle_year_has_expected_values_train)
print("Missing Count:", vehicle_year_missing_count_train)
print("Unique Values:", vehicle_year_unique_values_train)

print("\nTest VEHICLE_YEAR analysis:")
print("Unique Count:", vehicle_year_unique_count_test)
print("Has Expected Values (['after 2015', 'before 2015']):", vehicle_year_has_expected_values_test)
print("Missing Count:", vehicle_year_missing_count_test)
print("Unique Values:", vehicle_year_unique_values_test)

######################################################################################################################

# Analyze the 'MARRIED' variable in the train dataset
married_unique_count_train = train['MARRIED'].nunique()  # Count unique values
married_has_expected_values_train = train['MARRIED'].dropna().isin([0, 1]).all()  # Check if all values are binary (0/1)
married_missing_count_train = train['MARRIED'].isnull().sum()  # Count missing values
married_unique_values_train = train['MARRIED'].dropna().unique()  # Unique values in train dataset

# Analyze the 'MARRIED' variable in the test dataset
married_unique_count_test = test['MARRIED'].nunique()  # Count unique values
married_has_expected_values_test = test['MARRIED'].dropna().isin([0, 1]).all()  # Check if all values are binary (0/1)
married_missing_count_test = test['MARRIED'].isnull().sum()  # Count missing values
married_unique_values_test = test['MARRIED'].dropna().unique()  # Unique values in test dataset

# Display results
print("\nTrain MARRIED analysis:")
print("Unique Count:", married_unique_count_train)
print("Has Expected Values (Binary 0/1):", married_has_expected_values_train)
print("Missing Count:", married_missing_count_train)
print("Unique Values:", married_unique_values_train.tolist())

print("\nTest MARRIED analysis:")
print("Unique Count:", married_unique_count_test)
print("Has Expected Values (Binary 0/1):", married_has_expected_values_test)
print("Missing Count:", married_missing_count_test)
print("Unique Values:", married_unique_values_test.tolist())

######################################################################################################################

# Analyze the 'CHILDREN' variable in the train dataset
children_unique_count_train = train['CHILDREN'].nunique()  # Count unique values
children_has_expected_values_train = train['CHILDREN'].dropna().isin([0, 1]).all()  # Check if all values are binary (0/1)
children_missing_count_train = train['CHILDREN'].isnull().sum()  # Count missing values
children_unique_values_train = train['CHILDREN'].dropna().unique()  # Unique values in train dataset

# Analyze the 'CHILDREN' variable in the test dataset
children_unique_count_test = test['CHILDREN'].nunique()  # Count unique values
children_has_expected_values_test = test['CHILDREN'].dropna().isin([0, 1]).all()  # Check if all values are binary (0/1)
children_missing_count_test = test['CHILDREN'].isnull().sum()  # Count missing values
children_unique_values_test = test['CHILDREN'].dropna().unique()  # Unique values in test dataset

# Display results
print("\nTrain CHILDREN analysis:")
print("Unique Count:", children_unique_count_train)
print("Has Expected Values (Binary 0/1):", children_has_expected_values_train)
print("Missing Count:", children_missing_count_train)
print("Unique Values:", children_unique_values_train.tolist())

print("\nTest CHILDREN analysis:")
print("Unique Count:", children_unique_count_test)
print("Has Expected Values (Binary 0/1):", children_has_expected_values_test)
print("Missing Count:", children_missing_count_test)
print("Unique Values:", children_unique_values_test.tolist())

######################################################################################################################

# Analyze the 'POSTAL_CODE' variable in the train dataset
postal_unique_count_train = train['POSTAL_CODE'].nunique()  # Count unique values
postal_has_expected_values_train = train['POSTAL_CODE'].dropna().apply(lambda x: str(x).isdigit()).all()  # Check if all values are numeric and integers
postal_missing_count_train = train['POSTAL_CODE'].isnull().sum()  # Count missing values
postal_unique_values_train = train['POSTAL_CODE'].dropna().unique()  # Unique values in train dataset

# Analyze the 'POSTAL_CODE' variable in the test dataset
postal_unique_count_test = test['POSTAL_CODE'].nunique()  # Count unique values
postal_has_expected_values_test = test['POSTAL_CODE'].dropna().apply(lambda x: str(x).isdigit()).all()  # Check if all values are numeric and integers
postal_missing_count_test = test['POSTAL_CODE'].isnull().sum()  # Count missing values
postal_unique_values_test = test['POSTAL_CODE'].dropna().unique()  # Unique values in test dataset

# Display results
print("\nTrain POSTAL_CODE analysis:")
print("Unique Count:", postal_unique_count_train)
print("Has Expected Values (All Numeric and Integers):", postal_has_expected_values_train)
print("Missing Count:", postal_missing_count_train)
print("Unique Values:", sorted(postal_unique_values_train.tolist()))

print("\nTest POSTAL_CODE analysis:")
print("Unique Count:", postal_unique_count_test)
print("Has Expected Values (All Numeric and Integers):", postal_has_expected_values_test)
print("Missing Count:", postal_missing_count_test)
print("Unique Values:", sorted(postal_unique_values_test.tolist()))

######################################################################################################################

# Analyze the 'ANNUAL_MILEAGE' variable in the train dataset
annual_mileage_unique_count_train = train['ANNUAL_MILEAGE'].nunique()  # Count unique values
annual_mileage_has_expected_values_train = train['ANNUAL_MILEAGE'].dropna().apply(lambda x: float(x).is_integer()).all()  # Check if all values are numeric and integers
annual_mileage_missing_count_train = train['ANNUAL_MILEAGE'].isnull().sum()  # Count missing values
annual_mileage_unique_values_train = train['ANNUAL_MILEAGE'].dropna().unique()  # Unique values in train dataset

# Analyze the 'ANNUAL_MILEAGE' variable in the test dataset
annual_mileage_unique_count_test = test['ANNUAL_MILEAGE'].nunique()  # Count unique values
annual_mileage_has_expected_values_test = test['ANNUAL_MILEAGE'].dropna().apply(lambda x: float(x).is_integer()).all()  # Check if all values are numeric and integers
annual_mileage_missing_count_test = test['ANNUAL_MILEAGE'].isnull().sum()  # Count missing values
annual_mileage_unique_values_test = test['ANNUAL_MILEAGE'].dropna().unique()  # Unique values in test dataset

# Display results
print("\nTrain ANNUAL_MILEAGE analysis:")
print("Unique Count:", annual_mileage_unique_count_train)
print("Has Expected Values (All Numeric and Integers):", annual_mileage_has_expected_values_train)
print("Missing Count:", annual_mileage_missing_count_train)
print("Unique Values:", sorted(annual_mileage_unique_values_train.tolist()))


print("\nTest ANNUAL_MILEAGE analysis:")
print("Unique Count:", annual_mileage_unique_count_test)
print("Has Expected Values (All Numeric and Integers):", annual_mileage_has_expected_values_test)
print("Missing Count:", annual_mileage_missing_count_test)
print("Unique Values:", sorted(annual_mileage_unique_values_test.tolist()))

######################################################################################################################

# Analyze the 'VEHICLE_TYPE' variable in the train dataset
vehicle_type_unique_count_train = train['VEHICLE_TYPE'].nunique()  # Count unique values
vehicle_type_has_expected_values_train = all(
    val in ['sedan', 'sports car'] for val in train['VEHICLE_TYPE'].dropna().unique()
)  # Check if all values are in the expected list
vehicle_type_missing_count_train = train['VEHICLE_TYPE'].isnull().sum()  # Count missing values
vehicle_type_unique_values_train = train['VEHICLE_TYPE'].dropna().unique()  # Unique values in train dataset

# Analyze the 'VEHICLE_TYPE' variable in the test dataset
vehicle_type_unique_count_test = test['VEHICLE_TYPE'].nunique()  # Count unique values
vehicle_type_has_expected_values_test = all(
    val in ['sedan', 'sports car'] for val in test['VEHICLE_TYPE'].dropna().unique()
)  # Check if all values are in the expected list
vehicle_type_missing_count_test = test['VEHICLE_TYPE'].isnull().sum()  # Count missing values
vehicle_type_unique_values_test = test['VEHICLE_TYPE'].dropna().unique()  # Unique values in test dataset

# Display results
print("\nTrain VEHICLE_TYPE analysis:")
print("Unique Count:", vehicle_type_unique_count_train)
print("Has Expected Values (['sedan', 'sports car']):", vehicle_type_has_expected_values_train)
print("Missing Count:", vehicle_type_missing_count_train)
print("Unique Values:", vehicle_type_unique_values_train.tolist())

print("\nTest VEHICLE_TYPE analysis:")
print("Unique Count:", vehicle_type_unique_count_test)
print("Has Expected Values (['sedan', 'sports car']):", vehicle_type_has_expected_values_test)
print("Missing Count:", vehicle_type_missing_count_test)
print("Unique Values:", vehicle_type_unique_values_test.tolist())

######################################################################################################################

# Analyze the 'SPEEDING_VIOLATIONS' variable in the train dataset
speeding_unique_count_train = train['SPEEDING_VIOLATIONS'].nunique()  # Count unique values
speeding_has_expected_values_train = train['SPEEDING_VIOLATIONS'].dropna().apply(lambda x: float(x).is_integer()).all()  # Check if all values are numeric and integers
speeding_missing_count_train = train['SPEEDING_VIOLATIONS'].isnull().sum()  # Count missing values
speeding_unique_values_train = train['SPEEDING_VIOLATIONS'].dropna().unique()  # Unique values in train dataset

# Analyze the 'SPEEDING_VIOLATIONS' variable in the test dataset
speeding_unique_count_test = test['SPEEDING_VIOLATIONS'].nunique()  # Count unique values
speeding_has_expected_values_test = test['SPEEDING_VIOLATIONS'].dropna().apply(lambda x: float(x).is_integer()).all()  # Check if all values are numeric and integers
speeding_missing_count_test = test['SPEEDING_VIOLATIONS'].isnull().sum()  # Count missing values
speeding_unique_values_test = test['SPEEDING_VIOLATIONS'].dropna().unique()  # Unique values in test dataset

# Display results
print("\nTrain SPEEDING_VIOLATIONS analysis:")
print("Unique Count:", speeding_unique_count_train)
print("Has Expected Values (All Numeric and Integers):", speeding_has_expected_values_train)
print("Missing Count:", speeding_missing_count_train)
print("Unique Values:", sorted(speeding_unique_values_train.tolist()))

print("\nTest SPEEDING_VIOLATIONS analysis:")
print("Unique Count:", speeding_unique_count_test)
print("Has Expected Values (All Numeric and Integers):", speeding_has_expected_values_test)
print("Missing Count:", speeding_missing_count_test)
print("Unique Values:", sorted(speeding_unique_values_test.tolist()))

######################################################################################################################

# Analyze the 'PAST_ACCIDENTS' variable in the train dataset
past_accidents_unique_count_train = train['PAST_ACCIDENTS'].nunique()  # Count unique values
past_accidents_has_expected_values_train = train['PAST_ACCIDENTS'].dropna().apply(lambda x: float(x).is_integer()).all()  # Check if all values are numeric and integers
past_accidents_missing_count_train = train['PAST_ACCIDENTS'].isnull().sum()  # Count missing values
past_accidents_unique_values_train = train['PAST_ACCIDENTS'].dropna().unique()  # Unique values in train dataset

# Analyze the 'PAST_ACCIDENTS' variable in the test dataset
past_accidents_unique_count_test = test['PAST_ACCIDENTS'].nunique()  # Count unique values
past_accidents_has_expected_values_test = test['PAST_ACCIDENTS'].dropna().apply(lambda x: float(x).is_integer()).all()  # Check if all values are numeric and integers
past_accidents_missing_count_test = test['PAST_ACCIDENTS'].isnull().sum()  # Count missing values
past_accidents_unique_values_test = test['PAST_ACCIDENTS'].dropna().unique()  # Unique values in test dataset

# Display results
print("\nTrain PAST_ACCIDENTS analysis:")
print("Unique Count:", past_accidents_unique_count_train)
print("Has Expected Values (All Numeric and Integers):", past_accidents_has_expected_values_train)
print("Missing Count:", past_accidents_missing_count_train)
print("Unique Values:", sorted(past_accidents_unique_values_train.tolist()))

print("\nTest PAST_ACCIDENTS analysis:")
print("Unique Count:", past_accidents_unique_count_test)
print("Has Expected Values (All Numeric and Integers):", past_accidents_has_expected_values_test)
print("Missing Count:", past_accidents_missing_count_test)
print("Unique Values:", sorted(past_accidents_unique_values_test.tolist()))

######################################################################################################################


# Analyze the 'AGE' variable in the train dataset
age_unique_count_train = train['AGE'].nunique()  # Count unique values
age_has_expected_values_train = train['AGE'].dropna().apply(lambda x: float(x).is_integer() and 16 <= x <= 120).all()  # Check if all values are numeric, integers, and within range
age_missing_count_train = train['AGE'].isnull().sum()  # Count missing values
age_unique_values_train = sorted(train['AGE'].dropna().unique().tolist())  # Unique values sorted in ascending order

# Analyze the 'AGE' variable in the test dataset
age_unique_count_test = test['AGE'].nunique()  # Count unique values
age_has_expected_values_test = test['AGE'].dropna().apply(lambda x: float(x).is_integer() and 16 <= x <= 120).all()  # Check if all values are numeric, integers, and within range
age_missing_count_test = test['AGE'].isnull().sum()  # Count missing values
age_unique_values_test = sorted(test['AGE'].dropna().unique().tolist())  # Unique values sorted in ascending order

# Display results
print("\nTrain AGE analysis:")
print("Unique Count:", age_unique_count_train)
print("Has Expected Values (Numeric, Integers Between 16 and 120):", age_has_expected_values_train)
print("Missing Count:", age_missing_count_train)
print("Unique Values:", age_unique_values_train)

print("\nTest AGE analysis:")
print("Unique Count:", age_unique_count_test)
print("Has Expected Values (Numeric, Integers Between 16 and 120):", age_has_expected_values_test)
print("Missing Count:", age_missing_count_test)
print("Unique Values:", age_unique_values_test)


######################################################################################################################

# Analyze the 'DRIVING_EXPERIENCE' variable in the train dataset
driving_experience_unique_count_train = train['DRIVING_EXPERIENCE'].nunique()  # Count unique values
driving_experience_has_expected_values_train = train['DRIVING_EXPERIENCE'].dropna().apply(
    lambda x: float(x).is_integer() and 0 <= x <= 100
).all()  # Check if all values are numeric, integers, and within range
driving_experience_missing_count_train = train['DRIVING_EXPERIENCE'].isnull().sum()  # Count missing values
driving_experience_unique_values_train = sorted(train['DRIVING_EXPERIENCE'].dropna().unique().tolist())  # Unique values sorted in ascending order

# Analyze the 'DRIVING_EXPERIENCE' variable in the test dataset
driving_experience_unique_count_test = test['DRIVING_EXPERIENCE'].nunique()  # Count unique values
driving_experience_has_expected_values_test = test['DRIVING_EXPERIENCE'].dropna().apply(
    lambda x: float(x).is_integer() and 0 <= x <= 100
).all()  # Check if all values are numeric, integers, and within range
driving_experience_missing_count_test = test['DRIVING_EXPERIENCE'].isnull().sum()  # Count missing values
driving_experience_unique_values_test = sorted(test['DRIVING_EXPERIENCE'].dropna().unique().tolist())  # Unique values sorted in ascending order

# Display results
print("\nTrain DRIVING_EXPERIENCE analysis:")
print("Unique Count:", driving_experience_unique_count_train)
print("Has Expected Values (Numeric, Integers Between 0 and 100):", driving_experience_has_expected_values_train)
print("Missing Count:", driving_experience_missing_count_train)
print("Unique Values:", driving_experience_unique_values_train)

print("\nTest DRIVING_EXPERIENCE analysis:")
print("Unique Count:", driving_experience_unique_count_test)
print("Has Expected Values (Numeric, Integers Between 0 and 100):", driving_experience_has_expected_values_test)
print("Missing Count:", driving_experience_missing_count_test)
print("Unique Values:", driving_experience_unique_values_test)

######################################################################################################################

# Analyze the 'CLAIMS_INSURANCE_NEXT_YEAR' variable in the train dataset
claims_unique_count_train = train['CLAIMS_INSURANCE_NEXT_YEAR'].nunique()  # Count unique values
claims_has_expected_values_train = train['CLAIMS_INSURANCE_NEXT_YEAR'].dropna().isin([0, 1]).all()  # Check if all values are binary (0/1)
claims_missing_count_train = train['CLAIMS_INSURANCE_NEXT_YEAR'].isnull().sum()  # Count missing values
claims_unique_values_train = train['CLAIMS_INSURANCE_NEXT_YEAR'].dropna().unique()  # Unique values in train dataset

# Display results
print("\nTrain CLAIMS_INSURANCE_NEXT_YEAR analysis:")
print("Unique Count:", claims_unique_count_train)
print("Has Expected Values (Binary 0/1):", claims_has_expected_values_train)
print("Missing Count:", claims_missing_count_train)
print("Unique Values:", claims_unique_values_train.tolist())

######################################################################################################################

# Count the number of missing values for each column in the train dataset
print("\nMissing values in train dataset:")
for column, missing_count in train.isnull().sum().items():
    print(f"{column}: {missing_count}")

# Count the number of missing values for each column in the test dataset
print("\nMissing values in test dataset:")
for column, missing_count in test.isnull().sum().items():
    print(f"{column}: {missing_count}")

######################################################################################################################
# Calculate the number of missing values for each row in the train dataset
row_missing_counts = train.isnull().sum(axis=1)

# Count the number of rows for each possible number of missing values
missing_value_counts = row_missing_counts.value_counts().sort_index()

print("\nNumber of rows with x missing values in train dataset\n")
# Display the results
for missing_count, num_rows in missing_value_counts.items():
    print(f"Rows with {missing_count} missing values: {num_rows}")

######################################################################################################################
# Calculate the number of missing values for each row in the test dataset
row_missing_counts = test.isnull().sum(axis=1)

# Count the number of rows for each possible number of missing values
missing_value_counts = row_missing_counts.value_counts().sort_index()

print("\nNumber of rows with x missing values in test dataset\n")
# Display the results
for missing_count, num_rows in missing_value_counts.items():
    print(f"Rows with {missing_count} missing values: {num_rows}")


######################################################################################################################