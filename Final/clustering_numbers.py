import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

file_path = "cleaned data/processed_country_numbers.csv" 
df = pd.read_csv(file_path)

# Apply log transformation to GDP to reduce skewness
df['Log_GDP'] = np.log1p(df['GDP'])
features = ["Freedom Score", "Political Stability Index", "Log_GDP", "GINI"]
X = df[features]

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering using K-Means
k = 5  
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original DataFrame
df['Cluster'] = clusters

# View clusters
print("\nSample Countries from Each Cluster:")
for cluster in range(k):
    cluster_data = df[df['Cluster'] == cluster]
    sample_countries = cluster_data.sample(n=min(10, len(cluster_data)), random_state=42)
    print(f"\nCluster {cluster}:")
    print(sample_countries[['Country', 'Freedom Score', 'Political Stability Index', 'GDP', 'GINI']])

cluster_summary = df.groupby('Cluster')[["Freedom Score", "Political Stability Index", "Log_GDP", "GINI"]].mean()
print("\nCluster Summary:")
print(cluster_summary)

# Reducuction for PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# Plot the clusters with country name labels
plt.figure(figsize=(15, 10))
for cluster in range(k):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f"Cluster {cluster}")
    for i in cluster_data.index:
        plt.text(cluster_data['PCA1'][i], cluster_data['PCA2'][i], 
                 df['Country'][i], fontsize=8, alpha=0.7)

plt.title("Clusters of Countries")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend()
plt.grid()
plt.show()
