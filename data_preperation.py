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

df_t.drop(columns=['ID', 'VEHICLE_TYPE'], inplace=True)




# --- Identify and Remove Illogical Rows ---
# Rows with illogical values and missing critical data
illogical_rows = df_t[
    (df_t['AGE'] - df_t['DRIVING_EXPERIENCE'] < 16) &
    (pd.isna(df_t['ANNUAL_MILEAGE']) & pd.isna(df_t['CREDIT_SCORE']))
]
print(f"Number of illogical rows removed: {len(illogical_rows)}")
df_t.drop(index=illogical_rows.index, inplace=True)



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


df_t['SPEEDING_VIOLATIONS'] = pd.cut(df_t['SPEEDING_VIOLATIONS'], bins=[0, 1, 3,5,  float('inf')], labels=[0, 1, 2, 3], right=False)
print("Binned SPEEDING_VIOLATIONS into categories.")
#--- Create New Categories ---
#Combine "MARRIED" and "CHILDREN" into a single "FAMILY_STATUS" feature
df_t['FAMILY_STATUS'] = df_t['MARRIED'] * 1 + df_t['CHILDREN'] * 2
print("Created new feature: FAMILY_STATUS")


# --- Add New Features ---
# Create a weighted "DRIVING_EXPERIENCE" normalized by "AGE"
df_t['NORM_AGE_EXP_MEAN'] = (
    ((df_t['AGE'] - 16) / (80 - 16)) * 40 + df_t['DRIVING_EXPERIENCE']
) / 2
print("Added new feature: NORM_AGE_EXP_MEAN.")

# Drop redundant columns
df_t.drop(columns=['AGE', 'CHILDREN', 'MARRIED','DRIVING_EXPERIENCE'], inplace=True)
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
    'NORM_AGE_EXP_MEAN': 'Normalized Age and Experience Mean',
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
plt.title('Correlation Matrix (After Cleaning) and discret')

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


# 5. Histogram for SPEEDING_VIOLATIONS after transformation
plt.figure(figsize=(10, 6))
sns.histplot(df_t['SPEEDING_VIOLATIONS'], kde=True, color='purple', bins=22, edgecolor='black')
plt.title('Distribution of Speeding Violations (After Transformation)')
plt.xlabel('Speeding Violations')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('speeding_violations_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
plt.savefig('speeding_violations_distribution.png', dpi=300, bbox_inches='tight')


# 6. Pairplot for selected features
selected_features = ['CREDIT_SCORE', 'ANNUAL_MILEAGE', 'SPEEDING_VIOLATIONS', 'PAST_ACCIDENTS', 'CLAIMS_INSURANCE_NEXT_YEAR']
sns.pairplot(df_t[selected_features], hue='CLAIMS_INSURANCE_NEXT_YEAR', palette='coolwarm')
plt.suptitle('Pairplot of Selected Features', y=1.02)
plt.tight_layout()
plt.show()
plt.savefig('pairplot_selected_features.png', dpi=300, bbox_inches='tight')

# Calculate Spearman correlation for the cleaned DataFrame
spearman_corr_table_cleaned = df_t.corr(method='spearman')


# Plot the Spearman correlation heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(spearman_corr_table_cleaned, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Spearman Rank Correlation Matrix (After Cleaning)')
plt.tight_layout()
plt.savefig('spearman_correlation_matrix_after_cleaning.png', dpi=300, bbox_inches='tight')
plt.show()


