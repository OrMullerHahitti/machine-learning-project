from random import randint, uniform

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, f1_score
import graphviz

from part_two.data_preparations_part2 import prepare_data

# Step 1: Prepare the data
X_train, y_train, X_test, y_test = prepare_data(dummies=True)
base_estimator = DecisionTreeClassifier(max_depth=1,criterion='entropy')

# Step 2: Scale the data to avoid huge PCA ranges
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Initialize and train AdaBoost
ada = AdaBoostClassifier(estimator=base_estimator,n_estimators=50, random_state=42)
ada.fit(X_train, y_train)

# Step 4: Cross-validation
scores = cross_val_score(ada, X_train, y_train, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Mean CV score: {scores.mean():.4f}")

# Step 5: first evaluation of the model
y_pred_train = ada.predict(X_train)
y_pred_test = ada.predict(X_test)

f1_train = f1_score(y_train, y_pred_train)
f1_test = f1_score(y_test, y_pred_test)

print(f"F1 Training Score: {f1_train:.4f}")
print(f"F1 Testing Score: {f1_test:.4f}")
feature_names = X_train.columns.tolist()
# visualization for understanding the model
def plot_trees_with_weights(ada_model, max_trees=5, trees_per_row=3, feature_names=None):
    # Get the weights (estimator_weights_) from the AdaBoost model
    weights = ada_model.estimator_weights_

    # Determine the indices of the trees to plot
    n_estimators = len(ada_model.estimators_)
    indices = np.linspace(0, n_estimators - 1, max_trees, dtype=int)

    # Create a figure with subplots
    n_rows = (max_trees + trees_per_row - 1) // trees_per_row  # Calculate the number of rows needed
    fig, axes = plt.subplots(n_rows, trees_per_row, figsize=(20, 4 * n_rows))

    # Ensure axes is always a 1D array
    if n_rows == 1:
        axes = np.array([axes])

    # Plot the weights
    fig_weights, ax_weights = plt.subplots(figsize=(15, 5))
    ax_weights.bar(range(len(weights)), weights)
    ax_weights.set_title('Estimator Weights in AdaBoost')
    ax_weights.set_xlabel('Tree Index')
    ax_weights.set_ylabel('Weight (α)')

    # Plot trees with their weights in the title
    for i, (index, ax) in enumerate(zip(indices, axes.flatten())):
        plot_tree(ada_model.estimators_[index],
                  feature_names=feature_names,
                  class_names=['0', '1'],  # adjust based on your classes
                  filled=True,
                  ax=ax)
        ax.set_title(f'Tree {index}\nWeight: {weights[index]:.4f}')

    # Hide any unused subplots
    for j in range(i + 1, len(axes.flatten())):
        axes.flatten()[j].axis('off')

    plt.tight_layout()
    plt.show()

    # Print the weights for all trees
    print("\nAll Tree Weights:")
    for i, weight in enumerate(weights):
        print(f"Tree {i}: {weight:.4f}")

# Call the function
plot_trees_with_weights(ada, max_trees=6, trees_per_row=3, feature_names=feature_names)
#Step 6: evaluating after tuning
param_distributions = {
    'n_estimators': [50,100,150,200,250,300,350,400,450,500],          # Uniform random integers between 50 and 300
    'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5, 1.0],       # Uniform distribution between 0.01 and 1.01
    'estimator__max_depth': [1,2,3,4,5]    # Uniform random integers between 1 and 4
}

rand_search = RandomizedSearchCV(
    estimator=AdaBoostClassifier(estimator=base_estimator, random_state=42),
    param_distributions=param_distributions,
    n_iter=20,               # Number of parameter settings sampled
    scoring='f1',
    cv=5,
    random_state=42,
    n_jobs=-1
)
rand_search.fit(X_train, y_train)
print("Best parameters:", rand_search.best_params_)

best_model = rand_search.best_estimator_
y_pred_test = best_model.predict(X_test)
F1_train = f1_score(y_train, y_pred_train)
F1_test = f1_score(y_test, y_pred_test)
print("Best F1_train score:", F1_train)
print("Best F1_test score:", F1_test)

num_weak_learners_to_visualize = 6
selected_weak_learners = np.linspace(0, len(best_model.estimators_) - 1, num_weak_learners_to_visualize, dtype=int)

'''
 Function to visualize a weak learner as a tree from the new model
'''

# Function to visualize a weak learner as a tree
def visualize_tree_matplotlib(tree, feature_names, title):
    plt.figure(figsize=(20,10))
    plot_tree(tree, feature_names=feature_names, class_names=['0', '1'], filled=True, rounded=True)
    plt.title(title)
    plt.show()

# Select weak learners to visualize



# Visualize the selected weak learners as trees using matplotlib
for i, learner_index in enumerate(selected_weak_learners):
    visualize_tree_matplotlib(best_model.estimators_[learner_index], X_train.columns, f"Weak Learner {learner_index + 1}")


# Step 5: Apply PCA (2D) for visualization
pca = PCA(n_components=2)
X_train_2D = pca.fit_transform(X_train_scaled)


# Step 6: Define helper function to plot 2D boundaries
def plot_decision_boundary_2D(classifier, X, y, pca_model, step_size=0.1):
    plt.figure(figsize=(10, 6))

    # Calculate bounds
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2

    # Use a bigger step size to reduce memory usage
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step_size),
        np.arange(y_min, y_max, step_size)
    )

    # Project the meshgrid back to original feature space
    grid_original = pca_model.inverse_transform(np.c_[xx.ravel(), yy.ravel()])

    # Predict
    Z = classifier.predict(grid_original).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.RdBu, s=20)
    plt.title("Decision Boundary of the Final Strong Classifier")
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()


# Plot the decision boundaries of the final strong classifier
plot_decision_boundary_2D(
    best_model,
    X_train_2D,
    y_train,
    pca_model=pca,
    step_size=0.02
)
