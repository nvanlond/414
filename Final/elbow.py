import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

file_path = "processed_country_numbers.csv"
df = pd.read_csv(file_path)

print("Checking for missing values in the dataset:")
print(df.isnull().sum())

# Log transformation for GDP
df['Log_GDP'] = np.log1p(df['GDP']) 

# Select the features for clustering
features = ["Freedom Score", "Political Stability Index", "Log_GDP", "GINI"] 
X = df[features]

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method
distortions = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    distortions.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, distortions, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Distortion (Inertia)")
plt.xticks(k_range)
plt.grid()
plt.show()
