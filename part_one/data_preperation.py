import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sem.logic import is_funcvar
from numpy.ma.core import not_equal
from pandas.core.interchange.dataframe_protocol import DataFrame
#from pyasn1_modules.rfc6031 import id_pskc_friendlyName

from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import zscore

# Import custom modules
from part_one.data_analysis import numeric_train
# Copy the numeric training data
corrs = {}
class correlation_analysis:
    def __init__(self,df:pd.DataFrame,feature1:str ,feature2:str,method = 'spearman'):
        self.method = method
        self.feature1 = feature1
        self.feature2 = feature2
        self.change = None
        self.corrs = {'before_cleaning': df.corr(method=self.method)[feature1][feature2]}
    def add_after_cleaning(self, df:pd.DataFrame, feature=None):
        if feature:
            self.change = feature
        self.corrs['after_cleaning'] = df.corr(method=self.method)[self.feature1 if not feature else feature][self.feature2]
    def print_corr(self,label_map:dict ):
        return (f"--------\n{label_map[self.feature1]} and {label_map[self.feature2]} correlation before cleaning: {self.corrs['before_cleaning']}"
                f"\n{label_map[self.feature1] if not self.change else label_map[self.change]} and {label_map[self.feature2]} correlation after {'cleaning' if not self.change else 'change'}: {self.corrs['after_cleaning']}\n--------")
    @staticmethod
    def print_corrs(corrs:dict,label_map:dict):
        for key in corrs:
            print(corrs[key].print_corr(label_map))
df_t = numeric_train.copy()
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
    'FAMILY_STATUS' : 'Family Status',
    'GENDER' : 'Gender',
    'EDUCATION' : 'Education',
    'VEHICLE_OWNERSHIP' : 'Vehicle Ownership',
    'VEHICLE_YEAR' : 'Vehicle Year',
    'MARRIED' : 'Married',
    'CHILDREN' : 'Children',
}
# --- Remove Unnecessary Features ---

df_t.drop(columns=['ID', 'VEHICLE_TYPE'], inplace=True)
print("Dropped unnecessary columns: ID, VEHICLE_TYPE.")

# --- creating correlation for before and after the cleaning ---
corrs['corr_mileage_claims']=correlation_analysis(numeric_train,'ANNUAL_MILEAGE','CLAIMS_INSURANCE_NEXT_YEAR')
corrs['corr_children_claims']=correlation_analysis(numeric_train,'CHILDREN','CLAIMS_INSURANCE_NEXT_YEAR')
corrs['corr_married_claims']=correlation_analysis(numeric_train,'MARRIED','CLAIMS_INSURANCE_NEXT_YEAR')
corrs['corr_driving_experience_claims']=correlation_analysis(numeric_train,'DRIVING_EXPERIENCE','CLAIMS_INSURANCE_NEXT_YEAR')
corrs['corr_age_claims']=correlation_analysis(numeric_train,'AGE','CLAIMS_INSURANCE_NEXT_YEAR')
corrs['corr_credit_claims']=correlation_analysis(numeric_train,'CREDIT_SCORE','CLAIMS_INSURANCE_NEXT_YEAR')
corrs['corr_speeding_claims']=correlation_analysis(numeric_train,'SPEEDING_VIOLATIONS','CLAIMS_INSURANCE_NEXT_YEAR')

# --- Identify and Remove Illogical Rows ---
print(f'Amount of illogical rows with age and driving experience',{len(numeric_train[numeric_train['AGE'] - numeric_train['DRIVING_EXPERIENCE'] < 16])})
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
corrs['corr_credit_claims'].add_after_cleaning(df_t)
# --- Handle Missing Data with Linear Regression ---
# Predict missing "ANNUAL_MILEAGE" using Linear Regression
missing_annual_mileage = df_t['ANNUAL_MILEAGE'].isna()

corrs['corr_mileage_claims'].add_after_cleaning(df_t)

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
corrs['corr_mileage_claims'].add_after_cleaning(df_t)

df_t['SPEEDING_VIOLATIONS_Copy']= df_t['SPEEDING_VIOLATIONS'].copy()
df_t['SPEEDING_VIOLATIONS'] = pd.cut(df_t['SPEEDING_VIOLATIONS'], bins=[0, 1, 3,5,  float('inf')], labels=[0, 1, 2, 3], right=False)
print("Binned SPEEDING_VIOLATIONS into categories.")
corrs['corr_speeding_claims'].add_after_cleaning(df_t)
#--- Create New Categories ---
#Combine "MARRIED" and "CHILDREN" into a single "FAMILY_STATUS" feature
df_t['FAMILY_STATUS'] = df_t['MARRIED'] * 1 + df_t['CHILDREN'] * 2
print("Created new feature: FAMILY_STATUS")

corrs['corr_children_claims'].add_after_cleaning(df_t, 'FAMILY_STATUS')
corrs['corr_married_claims'].add_after_cleaning(df_t, 'FAMILY_STATUS')
# --- Add New Features ---
# Create a weighted "DRIVING_EXPERIENCE" normalized by "AGE"
df_t['NORM_AGE_EXP_MEAN'] = (
    ((df_t['AGE'] - 16) / (80 - 16)) * 40 + df_t['DRIVING_EXPERIENCE']
) / 2
print("Added new feature: NORM_AGE_EXP_MEAN.")
corrs['corr_driving_experience_claims'].add_after_cleaning(df_t, 'NORM_AGE_EXP_MEAN')
corrs['corr_age_claims'].add_after_cleaning(df_t, 'NORM_AGE_EXP_MEAN')

correlation_analysis.print_corrs(corrs,label_mapping)
# Drop redundant columns
df_t.drop(columns=['AGE', 'CHILDREN', 'MARRIED','DRIVING_EXPERIENCE'], inplace=True)
print("Dropped redundant columns: AGE, CHILDREN, MARRIED.")

# --- Correlation Analysis ---
# Define continuous columns for correlation analysis
correlation_columns = [
    'CREDIT_SCORE', 'ANNUAL_MILEAGE', 'SPEEDING_VIOLATIONS',
    'PAST_ACCIDENTS', 'DRIVING_EXPERIENCE', 'NORM_AGE_EXP_MEAN'
]
last_column = df_t.pop('CLAIMS_INSURANCE_NEXT_YEAR')  # Remove the column
df_t['CLAIMS_INSURANCE_NEXT_YEAR'] = last_column

df_t_renamed = df_t.rename(columns=label_mapping)
## 1. Distribution of CREDIT_SCORE before and after imputation
plt.figure(figsize=(14, 6))


# Before imputation
plt.subplot(1, 2, 1)
sns.histplot(numeric_train['CREDIT_SCORE'], kde=True, color='blue', bins=30, edgecolor='black')
plt.title('Distribution of Credit Score (Before Imputation)')
plt.xlabel(label_mapping['CREDIT_SCORE'])
plt.ylabel('Frequency')

# After imputation
plt.subplot(1, 2, 2)
sns.histplot(df_t['CREDIT_SCORE'], kde=True, color='green', bins=30, edgecolor='black')
plt.title('Distribution of Credit Score (After Imputation)')
plt.xlabel(label_mapping['CREDIT_SCORE'])
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
plt.ylabel(label_mapping['ANNUAL_MILEAGE'])

# After imputation
plt.subplot(1, 2, 2)
sns.boxplot(y=df_t['ANNUAL_MILEAGE'], color='green')
plt.title('Annual Mileage (After Imputation)')
plt.ylabel(label_mapping['ANNUAL_MILEAGE'])

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
plt.tight_layout()
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
plt.xlabel(label_mapping['FAMILY_STATUS'])
plt.ylabel('Count')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Single, No Kids', 'Married, No Kids', 'Single, With Kids', 'Married, With Kids'])
plt.tight_layout()
plt.savefig('family_status_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Histogram for SPEEDING_VIOLATIONS after transformation
plt.figure(figsize=(10, 6))
sns.histplot(df_t['SPEEDING_VIOLATIONS'], kde=True, color='purple', bins=22, edgecolor='black')
plt.title('Distribution of Speeding Violations (After Transformation)')
plt.xlabel(label_mapping['SPEEDING_VIOLATIONS'])
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('speeding_violations_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Histogram for SPEEDING_VIOLATIONS before transformation
plt.figure(figsize=(10, 6))
sns.histplot(numeric_train['SPEEDING_VIOLATIONS'], kde=True, color='purple', bins=22, edgecolor='black')
plt.title('Distribution of Speeding Violations (After Transformation)')
plt.xlabel(label_mapping['SPEEDING_VIOLATIONS'])
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('speeding_violations_distribution_before.png', dpi=300, bbox_inches='tight')
plt.show()

"""
"""
# Ensure SPEEDING_VIOLATIONS_CATEGORY is created with right categories
df_t['SPEEDING_VIOLATIONS_CATEGORY'] = pd.cut(
    df_t['SPEEDING_VIOLATIONS_Copy'],
    bins=[-float('inf'), 0, 2, 4, float('inf')],
    labels=["No Violations", "Few Violations", "Multiple Violations", "High Number of Violations"],
    right=True
)

# Plotting the histogram for SPEEDING_VIOLATIONS_CATEGORY with counts and percentages
plt.figure(figsize=(10, 6))
ax = sns.countplot(
    x=df_t['SPEEDING_VIOLATIONS_CATEGORY'],
    palette='viridis',
    edgecolor='black'
)
# Adding counts and percentages above the bars
total = len(df_t)
for p in ax.patches:
    count = int(p.get_height())
    percentage = f"{(count / total) * 100:.1f}%"
    ax.annotate(f'{count}\n({percentage})', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10)
plt.title('Distribution of Speeding Violations Categories (After Transformation)', fontsize=14)
plt.xlabel('Speeding Violations Category', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('speeding_violations_distribution_after.png', dpi=300, bbox_inches='tight')
plt.show()
"""
"""
# Calculate Spearman correlation for the cleaned DataFrame
spearman_corr_table_cleaned = df_t_renamed.corr(method='spearman')

# Plot the Spearman correlation heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(spearman_corr_table_cleaned, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Spearman Rank Correlation Matrix (After Cleaning)')
plt.tight_layout()
plt.savefig('spearman_correlation_matrix_after_cleaning.png', dpi=300, bbox_inches='tight')
plt.show()


## Pearson correlation: continuous variables

non_categorial_columns = [
    'CREDIT_SCORE', 'ANNUAL_MILEAGE',
    'PAST_ACCIDENTS', 'NORM_AGE_EXP_MEAN']

# Update column names based on label_mapping and get olny non-categorial columns
correlation_columns_mapped = [label_mapping[col] for col in non_categorial_columns]
df_t_renamed_filtered = df_t_renamed[correlation_columns_mapped]

# Calculate Pearson correlation for the cleaned DataFrame
pearson_corr_table_cleaned = df_t_renamed_filtered.corr(method='pearson')

# Plot the Pearson correlation heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(pearson_corr_table_cleaned, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Pearson Correlation Matrix (Filtered Columns)')
plt.tight_layout()
plt.savefig('pearson_correlation_matrix_filtered_columns.png', dpi=300, bbox_inches='tight')
plt.show()



df_t_renamed.to_csv('../prepared_data/cleaned_data_new.csv', index=False)

