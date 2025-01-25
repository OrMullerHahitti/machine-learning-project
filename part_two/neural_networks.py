import pandas as pd
from data_preparations_part2 import data, prepare_data, normalize_dataset
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import itertools
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import seaborn as sns


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

# Default properties for each hyperparameter
hidden_layer_sizes = model.hidden_layer_sizes  # Default hidden layer sizes
max_iter = model.max_iter                      # Default maximum iterations
alpha = model.alpha                            # Default L2 regularization
learning_rate = model.learning_rate            # Default learning rate schedule
batch_size = model.batch_size                  # Default batch size
early_stopping = model.early_stopping          # Default early stopping behavior
activation_function = model.activation         # Default activation function

# Print the details for each hyperparameter
print(f"Hidden Layer Sizes (Default): {hidden_layer_sizes}  # Number of neurons in each hidden layer")
print(f"Max Iterations (Default): {max_iter}  # Maximum number of epochs over the training data")
print(f"Alpha (Default): {alpha}  # L2 regularization term to prevent overfitting")
print(f"Learning Rate (Default): {learning_rate}  # Strategy for updating learning rate during training")
print(f"Batch Size (Default): {batch_size}  # Number of samples per gradient update")
print(f"Early Stopping (Default): {early_stopping}  # Whether to stop training early when validation score stops improving")
print(f"Activation Function (Default): {activation_function}  # Activation function for the hidden layer neurons")
# Number of neurons in the input layer
num_input_neurons = X_train.shape[1]
print(f"Number of neurons in the input layer: {num_input_neurons}")

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


#----------------------------------------------------------------------------------------------------------------------
""" Hyperparameter Tuning with GridSearchCV: 
GridSearchCV with 10-fold StratifiedKFold cross-validation is employed to optimize the hyperparameters 
of an MLPClassifier, targeting the F1 score.
The tuned parameters are:
- hidden_layer_sizes: Combinations of 1-2 layers, with 30-50 neurons per layer (step 10), where the order of layers matters.
- max_iter: Number of epochs (50).
- alpha: L2 regularization term (0.0001).
- learning_rate: Constant learning rate schedule.
- batch_size: Minibatch sizes (32, 64, 128, 200).
- early_stopping: Enabled to halt training early if validation performance stops improving (set to True)

The data is standardized before training. The process evaluates all possible combinations of hyperparameters, reporting:
Best hyperparameters based on cross-validation F1 score.
F1 scores on the train and test datasets using the best model.
"""
"""
layer_size = range(1, 3)  # Number of hidden layers
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
    'max_iter': [50],
    'alpha': [0.0001],
    'learning_rate': ['constant'],
    'batch_size': [32, 64, 128, 200],
    'n_iter_no_change': [20],
    'early_stopping': [True]  # Include early stopping
}

# Use StratifiedKFold for cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    MLPClassifier(random_state=42),
    param_grid,
    scoring='f1',
    cv=cv,
    verbose=3,  # Verbose level for detailed progress
)

# Fit the model with Standardized data
grid_search.fit(X_train_standard, y_train)

# Best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best F1 score:", grid_search.best_score_)

# Evaluate on the train set
best_model = grid_search.best_estimator_
y_pred_train = best_model.predict(X_train_standard)
f1_train = f1_score(y_train, y_pred_train)
print("Train F1 Score:", f1_train)

# Evaluate on the test set
y_pred_test = best_model.predict(X_test_standard)
f1_test = f1_score(y_test, y_pred_test)
print("Test F1 Score:", f1_test)

"""
#----------------------------------------------------------------------------------------------------------------------
"""
The code evaluates hyperparameter tuning results from GridSearchCV,
 extracts the top 10 configurations based on the mean test F1 score,
  and computes their train and test F1 scores on Min-Max normalized data.
Results are saved to a CSV file for further analysis, highlighting the best-performing models and their configurations.
"""
"""
# Collect results from GridSearchCV
results_df = pd.DataFrame(grid_search.cv_results_)

# Save the results as a CSV file
results_csv_path = "hyperparameter_tuning_results.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"Hyperparameter tuning results saved to: {results_csv_path}")
"""
# Load the saved results CSV file
results_csv_path = "hyperparameter_tuning_results.csv"
results_df = pd.read_csv(results_csv_path)

# Rename columns for readability
results_df = results_df.rename(columns={
    'param_batch_size': 'Batch Size',
    'param_hidden_layer_sizes': 'Hidden Layer Sizes',
    'param_max_iter': 'Max Iter',
    'mean_test_score': 'Mean Test F1 Score'
})


# Sort by Mean Test F1 Score and extract the top 10 configurations
top_10_configurations = results_df.sort_values(by='Mean Test F1 Score', ascending=False).head(10)

# Prepare a DataFrame to store the results
detailed_results = []

# Evaluate Train and Test F1 Scores for each of the top 10 configurations
for idx, row in top_10_configurations.iterrows():
    # Extract parameters for the current configuration
    params = {
        'hidden_layer_sizes': eval(row['Hidden Layer Sizes']),
        'batch_size': int(row['Batch Size']),
        'max_iter': int(row['Max Iter']),
        'random_state': 42,
        'learning_rate': 'constant',
        'alpha': 0.0001,
        'early_stopping': True,
        'n_iter_no_change': 20
    }

    # Train the model with the current configuration
    model = MLPClassifier(**params)
    model.fit(X_train_standard, y_train)

    # Compute F1 scores on training and test sets
    y_pred_train = model.predict(X_train_standard)
    y_pred_test = model.predict(X_test_standard)
    f1_train = f1_score(y_train, y_pred_train)
    f1_test = f1_score(y_test, y_pred_test)

    # Add n_iter_ (number of iterations)
    n_iter = model.n_iter_

    # Store the results
    detailed_results.append({
        'Batch Size': params['batch_size'],
        'Hidden Layer Sizes': params['hidden_layer_sizes'],
        'n_iter': n_iter,  # Add n_iter to the results
        'Max Iter': params['max_iter'],
        'Train F1 Score': round(f1_train,6),
        'Test F1 Score': round(f1_test,6),
        'Mean Test F1 Score': round(row['Mean Test F1 Score'],6)  # Add Mean Test F1 Score for this config
    })

# Convert detailed results to a DataFrame
detailed_results_df = pd.DataFrame(detailed_results)
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)

# Display the detailed results
print("\nTop 10 Configurations with Train and Test F1 Scores:")
print(detailed_results_df)

# Optional: Save to a CSV file
detailed_results_csv_path = "detailed_hyperparameter_results.csv"
detailed_results_df.to_csv(detailed_results_csv_path, index=False)
print(f"\nDetailed hyperparameter results saved to: {detailed_results_csv_path}")

####################################################################################################
# Create a figure and an axis for the table
fig, ax = plt.subplots(figsize=(12, 6))  # Adjust the figure size as needed
ax.axis('tight')
ax.axis('off')  # Turn off the axes

# Create the table in the matplotlib figure
table = ax.table(
    cellText=detailed_results_df.values,  # Data for the table
    colLabels=detailed_results_df.columns,  # Column headers
    cellLoc='center',  # Align cell text to center
    loc='center'  # Position the table at the center of the figure
)

# Adjust the font size and column widths
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(detailed_results_df.columns))))

# Add padding above and below the text in the table
for key, cell in table.get_celld().items():
    cell.set_text_props(va='center', ha='center')  # Center-align vertically and horizontally
    cell.set_height(cell.get_height() + 0.05)  # Add extra height (adjust as needed)

# Save the table as an image file
export_path = "top_10_configurations_table.png"
plt.savefig(export_path, bbox_inches='tight', dpi=300)  # Save at high resolution
print(f"Table exported as image: {export_path}")

# Display the table image
plt.show()



# Rename columns for consistency (if needed)
results_df = results_df.rename(columns={
    'param_batch_size': 'Batch Size',
    'param_hidden_layer_sizes': 'Hidden Layer Sizes',
    'mean_test_score': 'Mean Test F1 Score'
})

# Ensure 'Hidden Layer Sizes' is a string for better grouping in the plot
results_df['Hidden Layer Sizes'] = results_df['Hidden Layer Sizes'].astype(str)

# Create a plot for each batch size
plt.figure(figsize=(12, 8))

for batch_size in results_df['Batch Size'].unique():
    subset = results_df[results_df['Batch Size'] == batch_size]
    plt.plot(subset['Hidden Layer Sizes'], subset['Mean Test F1 Score'], label=f'Batch Size {batch_size}', marker='o')

# Customize the plot
plt.title('Mean Test F1 Score vs. Hidden Layer Sizes for Different Batch Sizes', fontsize=16)
plt.xlabel('Hidden Layer Sizes', fontsize=14)
plt.ylabel('Mean Test F1 Score', fontsize=14)
plt.xticks(rotation=45, fontsize=10)
plt.legend(title='Batch Size', fontsize=12)
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()


#########################################################################################

# Rename columns for readability
results_df = results_df.rename(columns={
    'param_batch_size': 'Batch Size',
    'param_hidden_layer_sizes': 'Hidden Layer Sizes',
    'mean_test_score': 'Mean Test F1 Score'
})

# Pivot the data for the heatmap
heatmap_data = results_df.pivot_table(
    index='Hidden Layer Sizes',
    columns='Batch Size',
    values='Mean Test F1 Score'
)

# Create the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='coolwarm', cbar=True, linewidths=0.5)
plt.title('Mean Test F1 Score by Hidden Layer Sizes and Batch Size', fontsize=16)
plt.xlabel('Batch Size', fontsize=14)
plt.ylabel('Hidden Layer Sizes', fontsize=14)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()

# Show the heatmap
plt.show()


# Optimized Hyperparameter
params = {
    'hidden_layer_sizes': (40,40),
    'max_iter': 50,
    'alpha': 0.0001,
    'learning_rate': 'constant',
    'batch_size': 64,
    'n_iter_no_change': 20,
    'early_stopping': True  # Include early stopping
}

model = MLPClassifier(**params)
model.fit(X_train_standard, y_train)

def predict_Optimized_Hyperparameter_NN(X_test):
    model = MLPClassifier(**params)
    model.fit(X_train_standard, y_train)
    return model.predict(X_test)



