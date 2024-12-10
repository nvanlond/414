import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load processed data
data = pd.read_csv('/Users/niels/code/414/module 4/processed_data.csv')

# Amplify the weight of 'rating' by multiplying it
data['rating_weighted'] = data['rating'] * 10

# Select features for clustering (exclude 'product_id')
features = [col for col in data.columns if col not in ['product_id', 'rating']]  # Replace 'rating' with 'rating_weighted'
clustering_data = data[features]

# Scale the features for clustering
scaler = StandardScaler()
scaled_features = scaler.fit_transform(clustering_data)

# Perform K-Means clustering with k=6
kmeans = KMeans(n_clusters=6, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_features)

# Randomly sample 5 rows from each cluster
def sample(group, n=5):
    return group.sample(n=min(n, len(group)), random_state=42)

random_samples = data.groupby('Cluster').apply(sample).reset_index(drop=True)
print("\nRandom Samples from Each Cluster:\n", random_samples)

# Calculate and print average rating and average actual price for each cluster
cluster_summary = data.groupby('Cluster')[['actual_price', 'rating']].mean()
print("\nAverage Actual Price and Average Rating by Cluster:\n", cluster_summary)

# Perform PCA for visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_features)

# Plot the clusters
plt.figure(figsize=(10, 6))
for cluster in range(6):
    cluster_points = pca_data[data['Cluster'] == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')

plt.title('K-Means Clustering Visualization (k=6)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)
plt.show()
