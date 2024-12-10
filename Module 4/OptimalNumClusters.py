import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load processed data
data = pd.read_csv('/Users/niels/code/414/module 4/processed_data.csv')

# Select features for clustering (exclude 'product_id')
features = [col for col in data.columns if col != 'product_id']
clustering_data = data[features]

# Store inertia values for each k
inertia_values = []

for k in range(2, 31):  # Test cluster sizes from 2 to 30
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(clustering_data)
    inertia_values.append(kmeans.inertia_)

# Print the first few inertia values
print("First five inertia values:", inertia_values[:5])

# Plot the Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(range(2, 31), inertia_values, marker='o')
plt.title('K-Means Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()
