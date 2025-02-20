import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # for heatmap
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, f1_score

from data_preparations_part2 import prepare_data


# --------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------

X_train, y_train, x_test, y_test = prepare_data(dummies=True)
# 1) Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 2) PCA (2 components)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

# Keep the original feature names for interpretability
feature_names = X_train.columns

# --------------------------------------------------------------------
# Find top contributing features for PC1 & PC2
# --------------------------------------------------------------------
abs_loadings_pc1 = np.abs(pca.components_[0])
abs_loadings_pc2 = np.abs(pca.components_[1])
print(f'list of all the self values for pc1{abs_loadings_pc1}')
print(f'list of all the self values for pc2 {abs_loadings_pc2}')
# Indices of the features with the largest absolute weight
top_idx_pc1 = np.argmax(abs_loadings_pc1)
top_idx_pc2 = np.argmax(abs_loadings_pc2)

#----------------------------------------------
# Create a DataFrame with the feature names and loadings
#----------------------------------------------
feature_list = list(zip(feature_names, abs_loadings_pc1, abs_loadings_pc2))

# Create a DataFrame from the feature list
df_features = pd.DataFrame(feature_list, columns=['Feature', 'PC1 Loading', 'PC2 Loading'])

# Calculate the average of PC1 and PC2 loadings
df_features['Average Loading'] = df_features[['PC1 Loading', 'PC2 Loading']].mean(axis=1)

# Sort the DataFrame by the average loading
df_features_sorted = df_features.sort_values(by='Average Loading', ascending=False).drop_duplicates(subset='Feature')

# Print the sorted DataFrame as a table
print(df_features_sorted.to_string(index=False))
top_feature_pc1 = feature_names[top_idx_pc1]
top_feature_pc2 = feature_names[top_idx_pc2]

# Create descriptive axis labels
label_pc1 = f" most dominant feature in (PC1) - {top_feature_pc1} "
label_pc2 = f" most dominant feature in (PC2) - {top_feature_pc2} "

# --------------------------------------------------------------------
# KMeans (2 Clusters)
# --------------------------------------------------------------------
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_train_pca)

# Evaluate
sil = silhouette_score(X_train_pca, clusters)
ari = adjusted_rand_score(y_train, clusters)
f_1 = f1_score(y_train, clusters)

print(f"Silhouette Score: {sil:.3f}")
print(f"Adjusted Rand Index: {ari:.3f}")
print(f"F1 Score: {f_1:.3f}")

# --------------------------------------------------------------------
# Plot 1: Cluster Assignments vs. True Labels
# --------------------------------------------------------------------
plt.figure(figsize=(12, 5))

# -- Cluster Assignments --
plt.subplot(1, 2, 1)
plt.scatter(X_train_pca[:, 0],
            X_train_pca[:, 1],
            c=clusters,
            cmap='viridis',
            edgecolor='k',
            alpha=0.7)

# Add centroids to the cluster plot
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            s=200, c='red', marker='X',
            label='Centroids')

plt.title("KMeans Clusters")
plt.xlabel(label_pc1)
plt.ylabel(label_pc2)
plt.legend()

# -- True Labels (assuming y_train is categorical or integer-labeled) --
plt.subplot(1, 2, 2)
plt.scatter(X_train_pca[:, 0],
            X_train_pca[:, 1],
            c=y_train.values.ravel(),
            cmap='coolwarm',
            edgecolor='k',
            alpha=0.7)
plt.title("True Labels")
plt.xlabel(label_pc1)
plt.ylabel(label_pc2)

plt.tight_layout()
plt.show()

# --------------------------------------------------------------------
# Plot 2: Bar Chart of Cluster Membership
# --------------------------------------------------------------------
unique_clusters, counts = np.unique(clusters, return_counts=True)

plt.figure(figsize=(6, 4))
plt.bar(unique_clusters, counts, color='skyblue', edgecolor='k')
plt.title("Cluster Membership Counts (KMeans)")
plt.xlabel("Cluster Label")
plt.ylabel("Number of Samples")
for i, v in enumerate(counts):
    plt.text(unique_clusters[i] - 0.06, v + 0.5, str(v), color='black', fontweight='bold')
plt.show()

# --------------------------------------------------------------------
# Plot 3: Heatmap of Clusters vs. True Labels (if labels are discrete)
# --------------------------------------------------------------------
# Check that y_train isn't continuous or has too many unique values
# If it's truly discrete/categorical, you can do a crosstab.

ctab = pd.crosstab(pd.Series(y_train, name='True Label'),
                   pd.Series(clusters, name='Cluster'))
plt.figure(figsize=(6, 4))
sns.heatmap(ctab, annot=True, cmap='Blues', fmt='d')
plt.title("Clusters vs. True Labels")
plt.show()

print("all done")