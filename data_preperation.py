import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import zscore

# Import custom modules
from data_analysis import numeric_train

# Copy the numeric training data
df_t = numeric_train.copy()
label_mapping = {
    'CREDIT_SCORE': 'Credit Score',
    'POSTAL_CODE': 'Postal Code',
    'ANNUAL_MILEAGE': 'Annual Mileage',
    'SPEEDING_VIOLATIONS': 'Speeding Violations',
    'PAST_ACCIDENTS': 'Past Accidents',
    'AGE': 'Age',
    'DRIVING_EXPERIENCE': 'Driving Experience',
    'WEIGHTED_AGE': 'Weighted Age',
    'CLAIMS_INSURANCE_NEXT_YEAR': 'Claims Next Year',
    'INCOME': 'Income Category'
}
# --- Remove Unnecessary Features ---

df_t.drop(columns=['POSTAL_CODE', 'ID', 'VEHICLE_TYPE'], inplace=True)



# --- Mutual Information Analysis ---
# Evaluate how much "GENDER" contributes to predicting "CLAIMS_INSURANCE_NEXT_YEAR"
mi = mutual_info_regression(df_t[['GENDER']], df_t['CLAIMS_INSURANCE_NEXT_YEAR'], discrete_features=True)
print(f"Mutual Information (GENDER): {mi[0]}")

# --- Identify and Remove Illogical Rows ---
# Rows with illogical values and missing critical data
illogical_rows = df_t[
    (df_t['AGE'] - df_t['DRIVING_EXPERIENCE'] < 16) &
    (pd.isna(df_t['ANNUAL_MILEAGE']) | pd.isna(df_t['CREDIT_SCORE']))
]
print(f"Number of illogical rows removed: {len(illogical_rows)}")
df_t.drop(index=illogical_rows.index, inplace=True)


# --- Add New Features ---
# Create a weighted "DRIVING_EXPERIENCE" normalized by "AGE"
df_t['WEIGHTED_AGE'] = (
    ((df_t['AGE'] - 16) / (80 - 16)) * 40 + df_t['DRIVING_EXPERIENCE']
) / 2
print("Added new feature: WEIGHTED_AGE")

# --- Handle Missing Data with K-Nearest Neighbors ---
# Impute missing "CREDIT_SCORE" values using "AGE" as a predictor
imputer = KNNImputer(n_neighbors=5, weights='distance')
subset_credit_age = df_t[['CREDIT_SCORE', 'AGE']]
df_t[['CREDIT_SCORE', 'AGE']] = imputer.fit_transform(subset_credit_age)
print("Missing CREDIT_SCORE values imputed using KNN.")

# --- Handle Missing Data with Linear Regression ---
# Predict missing "ANNUAL_MILEAGE" using Linear Regression
missing_annual_mileage = df_t['ANNUAL_MILEAGE'].isna()

# Split data into training and testing sets for imputation
train_data = df_t[df_t['ANNUAL_MILEAGE'].notna()]
test_data = df_t[missing_annual_mileage]

# Features and target for training
x_train = train_data[['CHILDREN', 'MARRIED', 'SPEEDING_VIOLATIONS']]
y_train = train_data['ANNUAL_MILEAGE']

# Train the Linear Regression model
reg = LinearRegression()
reg.fit(x_train, y_train)

# Predict missing values
x_test = test_data[['CHILDREN', 'MARRIED', 'SPEEDING_VIOLATIONS']]
y_pred = reg.predict(x_test)

df_t.loc[missing_annual_mileage, 'ANNUAL_MILEAGE'] = y_pred
print("Missing ANNUAL_MILEAGE values imputed using Linear Regression.")

df_t['INCOME'] = pd.cut(df_t['INCOME'], bins=[0, 30000, 60000, 100000, float('inf')], labels=['Low', 'Medium', 'High', 'Very High'], right=False)
income_mapping = {'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3}
df_t['INCOME'] = df_t['INCOME'].map(income_mapping)
# --- Create New Categories ---
# Combine "MARRIED" and "CHILDREN" into a single "FAMILY_STATUS" feature
df_t['FAMILY_STATUS'] = df_t['MARRIED'] * 1 + df_t['CHILDREN'] * 2
print("Created new feature: FAMILY_STATUS")

# Drop redundant columns
df_t.drop(columns=['AGE', 'CHILDREN', 'MARRIED'], inplace=True)
print("Dropped redundant columns: AGE, CHILDREN, MARRIED.")

# --- Correlation Analysis ---
# Define continuous columns for correlation analysis
correlation_columns = [
    'CREDIT_SCORE', 'ANNUAL_MILEAGE', 'SPEEDING_VIOLATIONS',
    'PAST_ACCIDENTS', 'DRIVING_EXPERIENCE', 'WEIGHTED_AGE'
]
last_column = df_t.pop('CLAIMS_INSURANCE_NEXT_YEAR')  # Remove the column
df_t['CLAIMS_INSURANCE_NEXT_YEAR'] = last_column
# Define descriptive labels for the plots
label_mapping = {
    'CREDIT_SCORE': 'Credit Score',
    'POSTAL_CODE': 'Postal Code',
    'ANNUAL_MILEAGE': 'Annual Mileage',
    'SPEEDING_VIOLATIONS': 'Speeding Violations',
    'PAST_ACCIDENTS': 'Past Accidents',
    'AGE': 'Age',
    'DRIVING_EXPERIENCE': 'Driving Experience',
    'WEIGHTED_AGE': 'Weighted Age',
    'CLAIMS_INSURANCE_NEXT_YEAR': 'Claims Next Year',
    'INCOME': 'Income Category',
    'FAMILY_STATUS' : 'Family Status'
}

# 1. Distribution of CREDIT_SCORE before and after imputation
plt.figure(figsize=(14, 6))

# Before imputation
plt.subplot(1, 2, 1)
sns.histplot(numeric_train['CREDIT_SCORE'], kde=True, color='blue', bins=30, edgecolor='black')
plt.title('Distribution of Credit Score (Before Imputation)')
plt.xlabel('Credit Score')
plt.ylabel('Frequency')

# After imputation
plt.subplot(1, 2, 2)
sns.histplot(df_t['CREDIT_SCORE'], kde=True, color='green', bins=30, edgecolor='black')
plt.title('Distribution of Credit Score (After Imputation)')
plt.xlabel('Credit Score')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('credit_score_before_after.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Boxplot comparison for ANNUAL_MILEAGE before and after imputation
plt.figure(figsize=(14, 6))

# Before imputation
plt.subplot(1, 2, 1)
sns.boxplot(y=numeric_train['ANNUAL_MILEAGE'], color='blue')
plt.title('Annual Mileage (Before Imputation)')
plt.ylabel('Annual Mileage')

# After imputation
plt.subplot(1, 2, 2)
sns.boxplot(y=df_t['ANNUAL_MILEAGE'], color='green')
plt.title('Annual Mileage (After Imputation)')
plt.ylabel('Annual Mileage')

plt.tight_layout()
plt.savefig('annual_mileage_before_after.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Correlation heatmap before and after cleaning
plt.figure(figsize=(14, 6))

# Before cleaning

sns.heatmap(
    numeric_train.select_dtypes(include=['number']).corr(),
    annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
    xticklabels=[label_mapping.get(col, col) for col in numeric_train.select_dtypes(include=['number']).columns],
    yticklabels=[label_mapping.get(row, row) for row in numeric_train.select_dtypes(include=['number']).columns]
)

plt.title('Correlation Matrix (Before Cleaning)')
plt.show()
plt.figure(figsize=(14, 6))

# After cleaning
sns.heatmap(
    df_t.select_dtypes(include=['number']).corr(),
    annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
    xticklabels=[label_mapping.get(col, col) for col in df_t.select_dtypes(include=['number']).columns],
    yticklabels=[label_mapping.get(row, row) for row in df_t.select_dtypes(include=['number']).columns]
)
plt.title('Correlation Matrix (After Cleaning)')

plt.tight_layout()
plt.savefig('correlation_matrix_before_after.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Count plot for FAMILY_STATUS after transformation
plt.figure(figsize=(8, 6))
sns.countplot(x='FAMILY_STATUS', data=df_t, palette='pastel')
plt.title('Distribution of Family Status (After Transformation)')
plt.xlabel('Family Status')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Single, No Kids', 'Married, No Kids', 'Single, With Kids', 'Married, With Kids'])
plt.tight_layout()
plt.savefig('family_status_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
