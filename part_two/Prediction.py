import pandas as pd
import openpyxl
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from neural_networks import predict_Optimized_Hyperparameter_NN
from data_preparations_part2 import cast_dataframe_to_int, get_dummies, normalize_dataset

X_Test=pd.read_csv('../X_test.csv', encoding='latin1')

# Replace non-numeric values with numeric codes
X_Test['EDUCATION'] = X_Test['EDUCATION'].replace({
    'none': 0, 'high school': 1, 'university': 2})
X_Test['INCOME'] = X_Test['INCOME'].replace({
    'poverty': 0, 'working class': 1, 'middle class': 2, 'upper class': 3})
X_Test['GENDER'] = X_Test['GENDER'].replace({
    'female': 0, 'male': 1})
X_Test['VEHICLE_YEAR'] = X_Test['VEHICLE_YEAR'].replace({
    'before 2015': 0, 'after 2015': 1})
X_Test['VEHICLE_TYPE'] = X_Test['VEHICLE_TYPE'].replace({
    'sedan': 0, 'sports car': 1})

# Overriding illogical values with NaN
X_Test.loc[~X_Test['EDUCATION'].isin([0, 1, 2]), 'EDUCATION'] = None
X_Test.loc[~X_Test['INCOME'].isin([0, 1, 2, 3]), 'INCOME'] = None
X_Test.loc[~X_Test['GENDER'].isin([0, 1]), 'GENDER'] = None
X_Test.loc[~X_Test['VEHICLE_YEAR'].isin([0, 1]), 'VEHICLE_YEAR'] = None
X_Test.loc[~X_Test['VEHICLE_TYPE'].isin([0, 1]), 'VEHICLE_TYPE'] = None

X_Test.drop(columns=['ID', 'VEHICLE_TYPE'], inplace=True)

# --- Handle Missing Data of "CREDIT_SCORE" ---
# Impute missing "CREDIT_SCORE" values using "AGE" as a predictor
imputer = KNNImputer(n_neighbors=5, weights='distance')
subset_credit_age = X_Test[['CREDIT_SCORE', 'AGE']]
X_Test[['CREDIT_SCORE', 'AGE']] = imputer.fit_transform(subset_credit_age)
# Impute missing "SPEEDING_VIOLATIONS_CATEGORY" values using "AGE" as a predictor
X_Test['SPEEDING_VIOLATIONS_Copy']= X_Test['SPEEDING_VIOLATIONS'].copy()
X_Test['SPEEDING_VIOLATIONS'] = pd.cut(X_Test['SPEEDING_VIOLATIONS'], bins=[0, 1, 3,5,  float('inf')], labels=[0, 1, 2, 3], right=False)

# --- Handle Missing Data of "ANNUAL_MILEAGE" ---
# Predict missing "ANNUAL_MILEAGE" using Linear Regression
missing_annual_mileage = X_Test['ANNUAL_MILEAGE'].isna()
# Split data into training and testing sets for imputation
train_data = X_Test[X_Test['ANNUAL_MILEAGE'].notna()]
test_data = X_Test[missing_annual_mileage]
# Features and target for training
x_train = train_data[['CHILDREN', 'MARRIED', 'SPEEDING_VIOLATIONS']]
y_train = train_data['ANNUAL_MILEAGE']
# Train the Linear Regression model
reg = LinearRegression()
reg.fit(x_train, y_train)
# Predict missing values
x_test = test_data[['CHILDREN', 'MARRIED', 'SPEEDING_VIOLATIONS']]
y_pred = reg.predict(x_test)
X_Test.loc[missing_annual_mileage, 'ANNUAL_MILEAGE'] = y_pred

# --- Create New Variables ---
#Combine "MARRIED" and "CHILDREN" into a single "FAMILY_STATUS" feature
X_Test['FAMILY_STATUS'] = X_Test['MARRIED'] * 1 + X_Test['CHILDREN'] * 2
# Create a weighted "DRIVING_EXPERIENCE" normalized by "AGE"
X_Test['NORM_AGE_EXP_MEAN'] = (
    ((X_Test['AGE'] - 16) / (80 - 16)) * 40 + X_Test['DRIVING_EXPERIENCE']) / 2

X_Test.drop(columns=['AGE', 'CHILDREN', 'MARRIED','DRIVING_EXPERIENCE'], inplace=True)

# Define descriptive labels
label_mapping = {
    'CREDIT_SCORE': 'Credit Score',
    'POSTAL_CODE': 'Postal Code',
    'ANNUAL_MILEAGE': 'Annual Mileage',
    'SPEEDING_VIOLATIONS': 'Speeding Violations',
    'PAST_ACCIDENTS': 'Past Accidents',
    'NORM_AGE_EXP_MEAN': 'Normalized Age and Experience Mean',
    'INCOME': 'Income Category',
    'FAMILY_STATUS': 'Family Status',
    'GENDER': 'Gender',
    'EDUCATION': 'Education',
    'VEHICLE_OWNERSHIP': 'Vehicle Ownership',
    'VEHICLE_YEAR': 'Vehicle Year',
}


X_Test= X_Test.rename(columns=label_mapping)

X_Test = X_Test.drop('SPEEDING_VIOLATIONS_Copy', axis=1)
X_Test= cast_dataframe_to_int(X_Test)
X_Test= get_dummies(X_Test)

categorical_ordered_continuous_cols = ['Credit Score', 'Normalized Age and Experience Mean', 'Annual Mileage',
                                       'Past Accidents', 'Education', 'Income Category', 'Speeding Violations']
X_Test = normalize_dataset(X_Test, columns=categorical_ordered_continuous_cols, method='standard')

print(X_Test)

prediction = predict_Optimized_Hyperparameter_NN(X_Test)
df = pd.DataFrame(prediction, columns=["target"])

# Define the filename
filename = "carinsurance_G5_ytest.xlsx"

# Save the DataFrame to an Excel file
df.to_excel(filename, index=False)

print(f"The file '{filename}' has been created successfully.")