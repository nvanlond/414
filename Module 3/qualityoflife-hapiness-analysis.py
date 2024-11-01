import pandas as pd
from sklearn.metrics import pairwise_distances

# Load the quality of life data
quality_of_life_df = pd.read_csv('country stats/Quality of life index by countries 2020.csv')

# Load the happiness dataset
happiness_df = pd.read_csv('hapiness data/data-2019.csv')

# Get Finland's data from the quality of life dataset
finland_data = quality_of_life_df[quality_of_life_df['Country'] == 'Finland'].iloc[:, 1:].values

# Create a DataFrame to store distances
similarities = {}

# Calculate the cosine distances to Finland for each country
for index, row in quality_of_life_df.iterrows():
    if row['Country'] != 'Finland':
        country_data = row.iloc[1:].values.reshape(1, -1)  # Reshape to 2D array
        distance = pairwise_distances(finland_data.reshape(1, -1), country_data, metric='cosine')[0][0]
        similarities[row['Country']] = distance

# Convert to DataFrame and sort by similarity
similarity_df = pd.DataFrame(list(similarities.items()), columns=['Country', 'Distance'])
similarity_df = similarity_df.sort_values(by='Distance')

# Print the top 10 countries by Rank 2019 from the happiness dataset without index
print("\nTop 10 Countries by Happiness Rank 2019:")
top_ranked_countries = happiness_df.nsmallest(10, 'Rank 2019')
print(top_ranked_countries[['Country', 'Rank 2019']].to_string(index=False))

# Print the top 10 countries by similarity to Finland without index
print("Top 10 Countries Similar to Finland:")
print(similarity_df.head(10).to_string(index=False))