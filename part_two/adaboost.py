import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from part_two.data_preparations_part2 import prepare_data

# Step 1: Prepare the data
X_train, y_train, X_test, y_test = prepare_data()

# Step 2: Scale the data to avoid huge PCA ranges
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Initialize and train AdaBoost
ada = AdaBoostClassifier(n_estimators=50, random_state=42)
ada.fit(X_train_scaled, y_train)

# Step 4: Cross-validation
scores = cross_val_score(ada, X_train_scaled, y_train, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Mean CV score: {scores.mean():.4f}")

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


# Step 7: Plot every 5th weak learner + final strong classifier
n_weak_learners = len(ada.estimators_)
rows = (n_weak_learners // 5) + 1
fig, axes = plt.subplots(rows, 1, figsize=(8, 4 * rows))
axes = np.array(axes).ravel()

plot_index = 0
for i in range(0, n_weak_learners, 5):
    plot_decision_boundary_2D(
        ada.estimators_[i],
        X_train_2D,
        y_train,
        axes[plot_index],
        title=f"Weak Classifier {i + 1}",
        pca_model=pca,
        step_size=0.1  # Adjust step size
    )
    plot_index += 1

# Plot final strong classifier
plot_decision_boundary_2D(
    ada,
    X_train_2D,
    y_train,
    axes[-1],
    title="Final Strong Classifier",
    pca_model=pca,
    step_size=0.1
)

plt.tight_layout()
plt.show()

print("done")
