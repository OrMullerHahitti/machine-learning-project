from sklearn.ensemble import adaBoostClassifier
from sklearn.datasets import make_classification

from part_two.data_preparations_part2 import prepare_data

df = pd.read_csv('../prepared_data/cleaned_data.csv')
X_train, y_train, X_test, y_test = prepare_data(df)

