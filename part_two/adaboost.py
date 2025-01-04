from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.ensemble import AdaBoostClassifier
import graphviz

from part_two.data_preparations_part2 import prepare_data

# Step 1: Prepare the data
X_train, y_train, X_test, y_test = prepare_data()
base_estimator = DecisionTreeClassifier(max_depth=1)

# Step 2: Scale the data to avoid huge PCA ranges
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Initialize and train AdaBoost
ada = AdaBoostClassifier(n_estimators=50, random_state=42)
ada.fit(X_train, y_train)

# Step 4: Cross-validation
scores = cross_val_score(ada, X_train, y_train, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Mean CV score: {scores.mean():.4f}")








# Function to visualize a weak learner as a tree
def visualize_tree_matplotlib(tree, feature_names, title):
    plt.figure(figsize=(20,10))
    plot_tree(tree, feature_names=feature_names, class_names=['0', '1'], filled=True, rounded=True)
    plt.title(title)
    plt.show()

# Select weak learners to visualize
num_weak_learners_to_visualize = 3
selected_weak_learners = np.linspace(0, len(ada.estimators_) - 1, num_weak_learners_to_visualize, dtype=int)


# Visualize the selected weak learners as trees using matplotlib
for i, learner_index in enumerate(selected_weak_learners):
    visualize_tree_matplotlib(ada.estimators_[learner_index], X_train.columns, f"Weak Learner {learner_index + 1}")


# Step 5: Apply PCA (2D) for visualization
pca = PCA(n_components=2)
X_train_2D = pca.fit_transform(X_train_scaled)


# Step 6: Define helper function to plot 2D boundaries
def plot_decision_boundary_2D(classifier, X, y, ax, title, pca_model, step_size=0.1):
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

    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.RdBu, s=20)
    ax.set_title(title)




# Plot final strong classifier
plot_decision_boundary_2D(
    ada,
    X_train_2D,
    y_train,
    title="Final Strong Classifier",
    pca_model=pca,
    step_size=0.1
)

plt.tight_layout()
plt.show()

print("done")
