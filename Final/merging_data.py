import pandas as pd

# Load the CSV files into DataFrames
primary_df = pd.read_csv("country_numbers.csv", index_col="Country")  # Primary DataFrame indexed by Country
secondary_df = pd.read_csv("cleaned_gini.csv")  # Secondary DataFrame

# Check for unmatched countries
unmatched_countries = set(secondary_df['Country']) - set(primary_df.index)

# Print unmatched countries
if unmatched_countries:
    print("Unmatched countries in the secondary DataFrame:")
    print(unmatched_countries)

# Filter the secondary DataFrame to include only matched countries
matched_secondary_df = secondary_df[secondary_df['Country'].isin(primary_df.index)]

# Merge the matched dataframes
merged_df = primary_df.merge(matched_secondary_df.set_index('Country'), left_index=True, right_index=True, how='left')

# Save the merged data to a new CSV file
merged_df.to_csv("country_numbers.csv")

# Display the merged DataFrame
print("\nMerged DataFrame:")
print(merged_df)
