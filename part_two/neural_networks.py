from data_preparations_part2 import data, prepare_data, normalize_dataset
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold


# Define columns for processing
# Categorical ordered and continuous columns to normalize
categorical_ordered_continuous_cols = ['Credit Score', 'Normalized Age and Experience Mean', 'Annual Mileage',
                                       'Past Accidents', 'Education', 'Income Category', 'Speeding Violations']

# Categorical unordered columns for dummy encoding
categorical_unordered_cols = ['Postal Code', 'Family Status']

# Prepare the data (handle dummies for unordered categories)
X_train, y_train, X_test, y_test = prepare_data(df=data, dummies=True)

# Initialize the Neural Network (MLPClassifier) with default parameters
model = MLPClassifier(random_state=42)

#----------------------------------------------------------------------------------------------------------------------
# Prepare the data (Normalize with Min-Max Scaling)
X_train_minmax = normalize_dataset(X_train, columns=categorical_ordered_continuous_cols, method='minmax')
X_test_minmax = normalize_dataset(X_test, columns=categorical_ordered_continuous_cols, method='minmax')

# Train and evaluate with Min-Max normalization
# Train the model using Min-Max normalized data
model.fit(X_train_minmax, y_train)

# Predictions for Min-Max normalization
y_pred_train_minmax = model.predict(X_train_minmax)
y_pred_test_minmax = model.predict(X_test_minmax)

# Calculate F1 scores for training and testing
f1_train_minmax = f1_score(y_train, y_pred_train_minmax)
f1_test_minmax = f1_score(y_test, y_pred_test_minmax)

#----------------------------------------------------------------------------------------------------------------------
# Prepare the data (Normalize with Standardization)
X_train_standard = normalize_dataset(X_train, columns=categorical_ordered_continuous_cols, method='standard')
X_test_standard = normalize_dataset(X_test, columns=categorical_ordered_continuous_cols, method='standard')

# Train and evaluate with Standard normalization
# Train the model using Standard normalized data
model.fit(X_train_standard, y_train)

# Predictions for Standard normalization
y_pred_train_standard = model.predict(X_train_standard)
y_pred_test_standard = model.predict(X_test_standard)

# Calculate F1 scores for training and testing
f1_train_standard = f1_score(y_train, y_pred_train_standard)
f1_test_standard = f1_score(y_test, y_pred_test_standard)
#----------------------------------------------------------------------------------------------------------------------

# Print results for both normalization methods
print("Comparison of Normalization Methods:")
print("Min-Max Normalization:")
print(f"F1 Training Score: {f1_train_minmax:.4f}")
print(f"F1 Testing Score: {f1_test_minmax:.4f}")
print("Standard Normalization:")
print(f"F1 Training Score: {f1_train_standard:.4f}")
print(f"F1 Testing Score: {f1_test_standard:.4f}")

# Determine which normalization method to continue with
if f1_test_minmax > f1_test_standard:
    print("Min-Max Normalization performs better based on F1 Testing Score.")
else:
    print("Standard Normalization performs better based on F1 Testing Score.")



"""
Hyperparameter Tuning:
This section uses GridSearchCV with 3-fold cross-validation to tune key hyperparameters 
of MLPClassifier:
    hidden_layer_sizes: Defines the architecture of the neural network.
    activation: Specifies the activation function for the hidden layers (relu or tanh).
    solver: Determines the optimization algorithm (adam or sgd).
    alpha: Sets the L2 regularization parameter.
    learning_rate: Controls the learning rate schedule.
It evaluates the model on both Min-Max and Standard normalization, optimizing for F1 score
to handle class imbalance effectively. Final results include the best parameters and performance on the test set.
"""
"""

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'hidden_layer_sizes': [(50,), (100,)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.0001],
    'learning_rate': ['constant'],
}

# Use StratifiedKFold for cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    MLPClassifier(random_state=42, max_iter=300),
    param_grid,
    scoring='f1',
    cv=cv,
    verbose=1
)

# Fit the model with Min-Max normalized data
grid_search.fit(X_train_minmax, y_train)

# Best parameters and score
print("Best parameters (Min-Max):", grid_search.best_params_)
print("Best F1 score (Min-Max):", grid_search.best_score_)

# Evaluate on the test set
best_model = grid_search.best_estimator_
y_pred_test = best_model.predict(X_test_minmax)
accuracy_test = accuracy_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)
print("Test Accuracy (Min-Max):", accuracy_test)
print("Test F1 Score (Min-Max):", f1_test)

# Fit the model with Standard normalized data
grid_search.fit(X_train_standard, y_train)

# Best parameters and score for Standard
print("Best parameters (Standard):", grid_search.best_params_)
print("Best F1 score (Standard):", grid_search.best_score_)

# Evaluate on the test set
y_pred_test_standard = grid_search.best_estimator_.predict(X_test_standard)
accuracy_test_standard = accuracy_score(y_test, y_pred_test_standard)
f1_test_standard = f1_score(y_test, y_pred_test_standard)
print("Test Accuracy (Standard):", accuracy_test_standard)
print("Test F1 Score (Standard):", f1_test_standard)
"""