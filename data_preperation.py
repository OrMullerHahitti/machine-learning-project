from math import isnan

from fontTools.ttLib.tables.S_V_G_ import doc_index_entry_format_0
from fontTools.ttLib.tables.otTraverse import dfs_base_table
from matplotlib import pyplot as plt
import seaborn as sn
from pandas import isnull
from pandas.core.interchange.dataframe_protocol import DataFrame
from scipy.signal import correlate
from unicodedata import numeric
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from data_analysis import  numeric_train
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression

import pandas as pd

df_t= numeric_train.copy()
#checking how much gender contributes
mi = mutual_info_regression(df_t[['GENDER']], df_t['CLAIMS_INSURANCE_NEXT_YEAR'], discrete_features=True)
print(f"Mutual Information: {mi[0]}")
#rows to drop where data is missing alot.
illogical_rows = df_t[(df_t['AGE']-df_t['DRIVING_EXPERIENCE']<16)
                      &( pd.isna(df_t['ANNUAL_MILEAGE'])
                      | pd.isna(df_t['CREDIT_SCORE'])) ]

#adding a new weighted driving experience/normalized age
df_t['WEIGHTED_AGE'] = (
    ((df_t['AGE'] - 16) / (80 - 16))*40 +
    df_t['DRIVING_EXPERIENCE']
) / 2



# completing missing data in credit score based on age using k-nearest neighbours
imputer = KNNImputer(n_neighbors=5, weights='distance')

subset_credit_age = df_t[['CREDIT_SCORE','AGE']]

subset_credit_age= imputer.fit_transform(subset_credit_age)

df_t[['CREDIT_SCORE','AGE']]=subset_credit_age

'''
completing missing values using linear regression on all
'''

mileage_categories = []
missing_annual_mileage = df_t['ANNUAL_MILEAGE'].isna()

train_data = df_t[df_t['ANNUAL_MILEAGE'].notna()]
test_data = df_t[df_t['ANNUAL_MILEAGE'].isna()]

y_train = train_data[['ANNUAL_MILEAGE']]
x_train = train_data[['CHILDREN', 'MARRIED', 'SPEEDING_VIOLATIONS']]

reg = LinearRegression()
reg.fit(x_train, y_train)

# Predict missing values
X_test = test_data[['CHILDREN', 'MARRIED', 'SPEEDING_VIOLATIONS']]
y_pred = reg.predict(X_test)

df_t.loc[missing_annual_mileage,'ANNUAL_MILEAGE']=y_pred

#creating new category to kids and married


df_t["FAMILY_STATUS"]=df_t['MARRIED']*1 +df_t['CHILDREN']*2

'''linear regression end'''


#dropping the non-relevant features
df_t.drop(columns=['AGE','CHILDREN','MARRIED'], inplace=True)


#checking corelation again
pearson_cor_new=(df_t.corr(method='pearson'))

plt.figure(figsize=(14, 10))
sn.heatmap(pearson_cor_new, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Pearson Correlation after data management')
plt.tight_layout()
plt.show()