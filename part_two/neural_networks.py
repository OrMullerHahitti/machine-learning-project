from data_preparations_part2 import data, prepare_data, normalize_dataset
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
import itertools



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

hidden_layer_sizes = model.hidden_layer_sizes  # Number of neurons in each hidden layer
activation_function = model.activation         # Activation function used

# Number of neurons in the input layer
num_input_neurons = X_train.shape[1]  # X_train is your training data
print(f"Number of neurons in the input layer: {num_input_neurons}")
# Displaying the results
print(f"Number of hidden layers: {len(hidden_layer_sizes)}")
print(f"Number of neurons in each hidden layer: {hidden_layer_sizes}")
print(f"Activation function: {activation_function}")

#----------------------------------------------------------------------------------------------------------------------
"""
Hyperparameter Tuning with RandomizedSearchCV:
RandomizedSearchCV with 5-fold StratifiedKFold cross-validation tunes MLPClassifier hyperparameters:
    - hidden_layer_sizes: 1-3 layers, 30-50 neurons per layer (step 10), order matters.
    - alpha: L2 regularization (0.0001).
    - learning_rate: Constant.
    - max_iter: 100, 200.
Evaluates 20 random combinations on Min-Max normalized data, reporting:
    - Best hyperparameters.
    - F1 scores on train and test sets.
"""

# Parameters
layer_size = range(1, 4)  # Number of hidden layers
neuron_Amount = range(30, 51, 10)  # Number of neurons per layer

# Generate combinations where order matters
hidden_layer_combinations = []
for i in layer_size:
    combinations = list(itertools.product(neuron_Amount, repeat=i))
    hidden_layer_combinations.extend(combinations)

# Check results
print(hidden_layer_combinations)


# Hyperparameter tuning with GridSearchCV
param_grid = {
    'hidden_layer_sizes': hidden_layer_combinations,
    'max_iter': [100, 200],
    'alpha': [0.0001],
    'learning_rate': ['constant']
}

# Use StratifiedKFold for cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize GridSearchCV
random_search = RandomizedSearchCV(
    MLPClassifier(random_state=100),
    param_grid,
    scoring='f1',
    cv=cv,
    verbose=3,  # Verbose level for detailed progress
    n_iter=20,
    random_state=42
)

# Fit the model with Min-Max normalized data
random_search.fit(X_train_minmax, y_train)

# Best parameters and score
print("Best parameters:", random_search.best_params_)
print("Best F1 score:", random_search.best_score_)

# Evaluate on the train set
best_model = random_search.best_estimator_
y_pred_train = best_model.predict(X_train_minmax)
f1_train = f1_score(y_train, y_pred_train)
print("Train F1 Score:", f1_train)

# Evaluate on the test set
y_pred_test = best_model.predict(X_test_minmax)
f1_test = f1_score(y_test, y_pred_test)
print("Test F1 Score:", f1_test)


#----------------------------------------------------------------------------------------------------------------------
"""
This next section visualizes the results to analyze the impact of hyperparameters:
- A scatter plot shows F1 scores against hidden layer configurations.
- Normalized distributions compare F1 scores for two values of max_iter (100 and 200).
Results are saved to a CSV file for further analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

# Collect results from RandomizedSearchCV
results_df = pd.DataFrame(random_search.cv_results_)

# Save the results as a CSV file
results_csv_path = "hyperparameter_tuning_results.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"Hyperparameter tuning results saved to: {results_csv_path}")

# Visualize the results

# Graph 1: Mean test F1 score vs. hidden layer sizes
plt.figure(figsize=(12, 8))
plt.scatter(
    results_df["param_hidden_layer_sizes"].astype(str),
    results_df["mean_test_score"],
    alpha=0.7,
    color='red'  # Set points to red
)
plt.xticks(rotation=90, color='black')
plt.title("F1 Score vs Hidden Layer Configurations", color='black')
plt.xlabel("Hidden Layer Sizes", color='black')
plt.ylabel("Mean Test F1 Score", color='black')
plt.grid(True)
plt.tight_layout()
plt.show()

# Graph 2: Normalized distributions for F1 scores with max_iter 100 and 200
f1_scores_100 = results_df[results_df["param_max_iter"] == 100]["mean_test_score"]
f1_scores_200 = results_df[results_df["param_max_iter"] == 200]["mean_test_score"]

# Calculate means and standard deviations for each distribution
mean_100, std_100 = f1_scores_100.mean(), f1_scores_100.std()
mean_200, std_200 = f1_scores_200.mean(), f1_scores_200.std()

# Generate x values for the curves
x = np.linspace(min(f1_scores_100.min(), f1_scores_200.min()),
                max(f1_scores_100.max(), f1_scores_200.max()), 500)

# Plot the distributions
plt.figure(figsize=(10, 6))
plt.plot(x, norm.pdf(x, mean_100, std_100), label='Max Iterations: 100', color='red')
plt.plot(x, norm.pdf(x, mean_200, std_200), label='Max Iterations: 200', color='blue')
plt.title("Normalized Distributions of F1 Scores", color='black')
plt.xlabel("F1 Score", color='black')
plt.ylabel("Density", color='black')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
