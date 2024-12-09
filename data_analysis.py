import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from scipy.stats import spearmanr, pearsonr, chi2_contingency, pointbiserialr
from tkinter import Tk, filedialog

# Set a clean theme with custom styling
sn.set_theme(style='whitegrid', context='talk')

#reading file
root = Tk()
root.withdraw()
#filePath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

#train=pd.read_csv(filePath, encoding='latin1')

train=pd.read_csv("XY_train.csv", encoding='latin1')

# Renaming the column OUTCOME to CLAIMS_INSURANCE_NEXT_YEAR
train.rename(columns={'OUTCOME': 'CLAIMS_INSURANCE_NEXT_YEAR'}, inplace=True)


####################################################################################################################
#Question 1:

train['CLAIMS_INSURANCE_NEXT_YEAR'] = train['CLAIMS_INSURANCE_NEXT_YEAR'].replace({0: 'No', 1: 'Yes'})

# Plot the distribution of CLAIMS_INSURANCE_NEXT_YEAR - target variable
claims_insurance_plot = sn.countplot(
    x=train['CLAIMS_INSURANCE_NEXT_YEAR'],
    order=['No','Yes'],
    color='red'
)
# Add labels to bars
for bar, (count, pct) in zip(claims_insurance_plot.patches, zip(
    train['CLAIMS_INSURANCE_NEXT_YEAR'].value_counts(ascending=True).values,
    train['CLAIMS_INSURANCE_NEXT_YEAR'].value_counts(ascending=True, normalize=True).values * 100
)):
    claims_insurance_plot.annotate(f'{count} ({pct:.1f}%)',
                                   (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                   ha='center', va='bottom', fontsize=12, color='black')
# Customize plot design
claims_insurance_plot.set_xlabel('Claims Insurance Next Year', fontsize=14, labelpad=10)
claims_insurance_plot.set_ylabel('Count', fontsize=14, labelpad=10)
claims_insurance_plot.set_title('Distribution of Claims Insurance Next Year', fontsize=18, pad=15)
claims_insurance_plot.set_ylim(0, train['CLAIMS_INSURANCE_NEXT_YEAR'].value_counts().max() * 1.1)
# Show the plot
plt.tight_layout()
plt.show()


#Distribution for CREDIT_SCORE
sn.set(style='darkgrid')
credit_score_plot = sn.displot(
    train['CREDIT_SCORE'], bins=30, kde=True, color='red', kde_kws={'clip': (0, None), 'bw_adjust': 0.5}
)
credit_score_plot.set_axis_labels('Credit Score (0-1)', 'Frequency')
credit_score_plot.fig.suptitle('Distribution of Credit Score', fontsize=16)
plt.tight_layout()
plt.show()


#Distribution for ANNUAL_MILEAGE
sn.set(style='darkgrid')
annual_mileage_plot = sn.displot(
    train['ANNUAL_MILEAGE'], bins=train['ANNUAL_MILEAGE'].nunique(), kde=True, color='red')
annual_mileage_plot.set_axis_labels('Annual Mileage', 'Frequency')
annual_mileage_plot.fig.suptitle('Distribution of Annual Mileage', fontsize=16)
plt.tight_layout()
plt.show()


#Distribution for AGE
################################## First Option
sn.set(style='darkgrid')
age_plot = sn.displot(
    train['AGE'], bins=30, kde=True, color='red', kde_kws={'clip': (0, None), 'bw_adjust': 0.5}
)
age_plot.set_axis_labels('Age', 'Frequency')
age_plot.fig.suptitle('Distribution of Age', fontsize=16)
plt.tight_layout()
plt.show()

################################## Second Option
sn.set(style='dark')
age_plot = sn.displot(train['AGE'],  bins=train['AGE'].nunique(), kde=True, color='red')
plt.xlim(train['AGE'].min(), train['AGE'].max())
plt.ylim(0, train['AGE'].value_counts().max()*1.1)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age')
plt.tight_layout()
plt.show()


#Distribution for DRIVING_EXPERIENCE
sn.set(style='dark')
driving_experience_plot = sn.displot(train['DRIVING_EXPERIENCE'], bins=train['DRIVING_EXPERIENCE'].nunique(),
                                     kde=True, color='red')
plt.xlim(train['DRIVING_EXPERIENCE'].min(), train['DRIVING_EXPERIENCE'].max())
plt.ylim(0, train['DRIVING_EXPERIENCE'].value_counts().max()*1.1)
plt.xlabel('Driving Experience (years)')
plt.ylabel('Frequency')
plt.title('Distribution of Driving Experience')
plt.tight_layout()
plt.show()


#Distribution for PAST_ACCIDENTS
sn.set(style='dark')
past_accidents_plot = sn.displot(
    train['PAST_ACCIDENTS'], bins=15, kde=True, kde_kws={'bw_adjust': 1.4}, color='red')
plt.xlim(train['PAST_ACCIDENTS'].min(), train['PAST_ACCIDENTS'].max())
plt.ylim(0, train['PAST_ACCIDENTS'].value_counts().max()*1.1)
past_accidents_plot.set_axis_labels('Past Accidents', 'Frequency')
past_accidents_plot.fig.suptitle('Distribution of Past Accidents', fontsize=16)
plt.tight_layout()
plt.show()

"""
# Convert binned values to numeric
train['PAST_ACCIDENTS_BINNED_NUMERIC'] = train['PAST_ACCIDENTS'].apply(lambda x: 4 if x > 3 else x)

# Calculate Point-Biserial Correlation for original and binned
correlation_original, p_value_original = pointbiserialr(train['PAST_ACCIDENTS'], train['CLAIMS_INSURANCE_NEXT_YEAR'])
correlation_binned, p_value_binned = pointbiserialr(train['PAST_ACCIDENTS_BINNED_NUMERIC'], train['CLAIMS_INSURANCE_NEXT_YEAR'])

print(f"Point-Biserial Correlation (Original): {correlation_original:.3f}, p-value: {p_value_original:.3f}")
print(f"Point-Biserial Correlation (Binned): {correlation_binned:.3f}, p-value: {p_value_binned:.3f}")
"""

####################################################################################################################

#Distribution for VEHICLE_YEAR
vehicle_year_plot = sn.countplot(
    x=train['VEHICLE_YEAR'],
    order=['before 2015', 'after 2015'],
    color='red'
)

# Add labels directly
for bar in vehicle_year_plot.patches:
    count = int(bar.get_height())
    pct = (count / len(train)) * 100
    vehicle_year_plot.annotate(f'{count} ({pct:.1f}%)',
                                   (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                   ha='center', va='bottom', fontsize=10, color='black')

# Customize plot
vehicle_year_plot.set(title='Distribution of VEHICLE_YEAR', xlabel='VEHICLE_YEAR', ylabel='Count')
plt.tight_layout()
plt.show()

####################################################################################################################
# Distribution for EDUCATION
education_plot = sn.countplot(
    x=train['EDUCATION'],
    order=['none', 'high school', 'university'],
    color='red'
)

# Add labels directly
for bar in education_plot.patches:
    count = int(bar.get_height())
    pct = (count / len(train)) * 100
    education_plot.annotate(f'{count} ({pct:.1f}%)',
                            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            ha='center', va='bottom', fontsize=10, color='black')

# Customize plot
education_plot.set(title='Distribution of EDUCATION', xlabel='EDUCATION', ylabel='Count')
plt.tight_layout()
plt.show()

####################################################################################################################

# Distribution for INCOME
income_plot = sn.countplot(
    x=train['INCOME'],
    order=['poverty', 'working class', 'middle class', 'upper class'],
    color='red'
)

# Add labels directly
for bar in income_plot.patches:
    count = int(bar.get_height())
    pct = (count / len(train)) * 100
    income_plot.annotate(f'{count} ({pct:.1f}%)',
                         (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         ha='center', va='bottom', fontsize=10, color='black')

# Customize plot
income_plot.set(title='Distribution of INCOME', xlabel='INCOME', ylabel='Count')
plt.tight_layout()
plt.show()

####################################################################################################################

# Distribution for GENDER
gender_plot = sn.countplot(
    x=train['GENDER'],
    order=["female", "male"],
    color='red'
)

# Add labels directly
for bar in gender_plot.patches:
    count = int(bar.get_height())
    pct = (count / len(train)) * 100
    gender_plot.annotate(f'{count} ({pct:.1f}%)',
                         (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         ha='center', va='bottom', fontsize=10, color='black')

# Customize plot
gender_plot.set(title='Distribution of GENDER', xlabel='GENDER', ylabel='Count')
plt.tight_layout()
plt.show()

####################################################################################################################

# Distribution for VEHICLE_TYPE
vehicle_type_plot = sn.countplot(
    x=train['VEHICLE_TYPE'],
    order=["sedan", "sports car"],
    color='red'
)

# Add labels directly
for bar in vehicle_type_plot.patches:
    count = int(bar.get_height())
    pct = (count / len(train)) * 100
    vehicle_type_plot.annotate(f'{count} ({pct:.1f}%)',
                               (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                               ha='center', va='bottom', fontsize=10, color='black')

# Customize plot
vehicle_type_plot.set(title='Distribution of VEHICLE_TYPE', xlabel='VEHICLE_TYPE', ylabel='Count')
plt.tight_layout()
plt.show()



######################################################################################################################


# Changing non-numeric values to numeric values for EDUCATION, INCOME, GENDER, VEHICLE_YEAR, VEHICLE_TYPE, and CLAIMS_INSURANCE_NEXT_YEAR
numeric_train = train.copy()

# Replace non-numeric values with numeric codes
numeric_train['EDUCATION'] = numeric_train['EDUCATION'].replace({
    'none': 0, 'high school': 1, 'university': 2
})
numeric_train['INCOME'] = numeric_train['INCOME'].replace({
    'poverty': 0, 'working class': 1, 'middle class': 2, 'upper class': 3
})
numeric_train['GENDER'] = numeric_train['GENDER'].replace({
    'female': 0, 'male': 1
})
numeric_train['VEHICLE_YEAR'] = numeric_train['VEHICLE_YEAR'].replace({
    'before 2015': 0, 'after 2015': 1
})
numeric_train['VEHICLE_TYPE'] = numeric_train['VEHICLE_TYPE'].replace({
    'sedan': 0, 'sports car': 1
})
numeric_train['CLAIMS_INSURANCE_NEXT_YEAR'] = numeric_train['CLAIMS_INSURANCE_NEXT_YEAR'].replace({
    'No': 0, 'Yes': 1
})

# Overriding illogical values with NaN
numeric_train.loc[~numeric_train['EDUCATION'].isin([0, 1, 2]), 'EDUCATION'] = None
numeric_train.loc[~numeric_train['INCOME'].isin([0, 1, 2, 3]), 'INCOME'] = None
numeric_train.loc[~numeric_train['GENDER'].isin([0, 1]), 'GENDER'] = None
numeric_train.loc[~numeric_train['VEHICLE_YEAR'].isin([0, 1]), 'VEHICLE_YEAR'] = None
numeric_train.loc[~numeric_train['VEHICLE_TYPE'].isin([0, 1]), 'VEHICLE_TYPE'] = None
numeric_train.loc[~numeric_train['CLAIMS_INSURANCE_NEXT_YEAR'].isin([0, 1]), 'CLAIMS_INSURANCE_NEXT_YEAR'] = None

# Pearson correlation: continuous variables
continuous_columns = [
    'ID', 'CREDIT_SCORE', 'POSTAL_CODE', 'ANNUAL_MILEAGE',
    'SPEEDING_VIOLATIONS', 'PAST_ACCIDENTS', 'AGE', 'DRIVING_EXPERIENCE'
]

# Pearson correlation for the specified continuous columns
pearson_corr_table = numeric_train[continuous_columns].corr(method='pearson')
print(pearson_corr_table.to_string())

plt.figure(figsize=(14, 10))
sn.heatmap(pearson_corr_table, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Pearson Rank Correlation Matrix')
plt.tight_layout()
plt.show()

# Spearman correlation: categorical and continuous variables
spearman_corr_table = numeric_train.corr(method='spearman')
print(spearman_corr_table.to_string())

plt.figure(figsize=(14, 10))
sn.heatmap(spearman_corr_table, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Spearman Rank Correlation Matrix')
plt.tight_layout()
plt.show()

####################################################################################################################

# Examine AGE and DRIVING_EXPERIENCE Relations

# Pearson correlation between AGE and DRIVING_EXPERIENCE
pearson_corr =pearson_corr_table['AGE']['DRIVING_EXPERIENCE']
print("Pearson correlation between AGE and DRIVING_EXPERIENCE: "+ str(pearson_corr))

# Scatter plot of AGE vs DRIVING_EXPERIENCE
plt.figure(figsize=(10, 5))
sn.scatterplot(x='AGE', y='DRIVING_EXPERIENCE', data=numeric_train, color='red')
plt.xlabel('AGE')
plt.ylabel('DRIVING EXPERIENCE')
plt.title('Scatter Plot of AGE vs DRIVING EXPERIENCE')
plt.show()

####################################################################################################################

# Examine CREDIT_SCORE and INCOME Relations

# Spearman correlation between AGE and DRIVING_EXPERIENCE
spearman_corr =spearman_corr_table['CREDIT_SCORE']['INCOME']
print("Spearman correlation between CREDIT SCORE and INCOME: "+ str(spearman_corr))

# Box plot to show the relationship between CREDIT_SCORE and INCOME
sn.boxplot(
    x='INCOME',
    y='CREDIT_SCORE',
    data=numeric_train,
    showfliers=False,
    palette="Set2",
)


# Customize x-axis labels with manual mapping
plt.xticks(ticks=range(4), labels=['poverty', 'working class', 'middle class', 'upper class'])


plt.title('Box Plot of INCOME - CREDIT_SCORE')
plt.xlabel('INCOME')
plt.ylabel('CREDIT_SCORE')
plt.show()

####################################################################################################################

# Examine GENDER and PAST_ACCIDENTS Relations

# Spearman correlation between GENDER and PAST_ACCIDENTS
spearman_corr = spearman_corr_table['GENDER']['PAST_ACCIDENTS']
print("Spearman correlation between GENDER and PAST_ACCIDENTS: " + str(spearman_corr))

# Box plot to show the relationship between GENDER and PAST_ACCIDENTS
sn.boxplot(
    x='GENDER',
    y='PAST_ACCIDENTS',
    data=numeric_train,
    showfliers=True,
    palette="Set2"
)

# Customize x-axis labels with manual mapping
plt.xticks(ticks=[0, 1], labels=['Female', 'Male'])

# Add title and labels
plt.title('Box Plot of GENDER - PAST_ACCIDENTS')
plt.xlabel('GENDER')
plt.ylabel('PAST ACCIDENTS')
plt.show()

####################################################################################################################

## Appendix

explained_var = 'CLAIMS_INSURANCE_NEXT_YEAR'

# Continuous for analysis
continuous_vars = [
    'ID', 'CREDIT_SCORE', 'POSTAL_CODE', 'ANNUAL_MILEAGE',
    'SPEEDING_VIOLATIONS', 'PAST_ACCIDENTS', 'AGE', 'DRIVING_EXPERIENCE'
]

# Melt the DataFrame to long format for boxplot analysis
melted_df = pd.melt(
    numeric_train,
    id_vars=[explained_var],
    value_vars=continuous_vars,
    var_name='Explanatory',
    value_name='Value'
)

# Create box plots using catplot
g = sn.catplot(
    x=explained_var,
    y='Value',
    hue=explained_var,
    data=melted_df,
    kind='box',
    col='Explanatory',
    col_wrap=3,
    sharex=False,
    sharey=False,
    palette='Set2',
    height=3,  # Reduce height of each plot
    aspect=0.8,  # Reduce width of each plot
    showfliers=True,  # Show outliers
    legend=False
)

# Remove the 'Explanatory =' text from subplot titles
for ax in g.axes.flat:
    ax.set_title(ax.get_title().replace("Explanatory = ", ""))

# Adjust spacing between subplots
g.fig.subplots_adjust(wspace=0.3, hspace=5)

# Dynamically adjust x-axis labels to show 'No' and 'Yes'
g.set_xticklabels(['No', 'Yes'])

# Adjust the layout
plt.tight_layout()
plt.show()

####################################################################################################################