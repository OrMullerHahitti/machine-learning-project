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


# Plot the distribution of CLAIMS_INSURANCE_NEXT_YEAR - target variable
claims_insurance_plot = sn.countplot(
    x=train['CLAIMS_INSURANCE_NEXT_YEAR'],
    order=train['CLAIMS_INSURANCE_NEXT_YEAR'].value_counts(ascending=True).index,
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





