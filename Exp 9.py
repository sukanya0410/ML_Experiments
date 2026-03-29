# -----------------------------
# Experiment: Principal Component Analysis (PCA)
# -----------------------------

# Import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -----------------------------
# Step 0: Create Sample Dataset
# (Instead of external CSV)
# -----------------------------
data = {
    'Feature1': [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2.0, 1.0, 1.5, 1.1],
    'Feature2': [2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9]
}

df = pd.DataFrame(data)

print("Original Data:\n", df)

# -----------------------------
# Step 1: Standardize Dataset
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

print("\nStandardized Data:\n", X_scaled)

# -----------------------------
# Step 2: Apply PCA
# -----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# -----------------------------
# Step 3: Results
# -----------------------------
print("\nPCA Transformed Data:\n", X_pca)

# -----------------------------
# Step 4: Variance Explained
# -----------------------------
print("\nExplained Variance Ratio:\n", pca.explained_variance_ratio_)

# -----------------------------
# Step 5: Eigenvalues (Optional)
# -----------------------------
print("\nEigenvalues:\n", pca.explained_variance_)