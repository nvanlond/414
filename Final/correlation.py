import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("merged_leader_country_data.csv")

# Group by Community and Cluster, and count the number of countries
grouped = df.groupby(['Community', 'Cluster'])['Country'].agg(['size', lambda x: '\n'.join(x)]).reset_index()
grouped.columns = ['Community', 'Cluster', 'Count', 'Countries']

print("Summary of Countries by Community and Cluster:\n")
for _, row in grouped.iterrows():
    print(f"Community {row['Community']} - Cluster {row['Cluster']}:")
    print(f"  Number of Countries: {row['Count']}")
    
plt.figure(figsize=(12, 8))

# Scatter plot
for i, row in grouped.iterrows():
    plt.scatter(
        row['Community'],  # x-axis: network community
        row['Cluster'],    # y-axis: socioeconomic cluster
        s=row['Count'] * 100,  # Bubble size
        alpha=0.7,
        label=f"Community {row['Community']}, Cluster {row['Cluster']}"
    )
    plt.text(
        row['Community'], row['Cluster'],
        row['Countries'],
        fontsize=8, alpha=0.9, ha='center', va='center'
    )

# Show plot
plt.title("Network Community vs Socioeconomic Cluster Distribution", fontsize=16)
plt.xlabel("Network Community", fontsize=14)  # Updated X-axis label
plt.ylabel("Socioeconomic Cluster", fontsize=14)  # Updated Y-axis label
plt.xticks(sorted(grouped['Community'].unique()))
plt.yticks(sorted(grouped['Cluster'].unique()))
plt.grid(alpha=0.3)

plt.xlim(grouped['Community'].min() - .75, grouped['Community'].max() + .75)
plt.ylim(grouped['Cluster'].min() - .75, grouped['Cluster'].max() + .75)

plt.tight_layout()
plt.show()
