import pandas as pd

# Used to merge network data communitites with socioeconomic clusters
df_leaders = pd.read_csv('module 2 and final data resuls/df_leaders.csv')
df_countries = pd.read_csv('module 2 and final data resuls/df_countries.csv')

df_leaders = df_leaders.drop(columns='Leader')

df_leaders['Country'] = df_leaders['Country'].str.strip().str.title()
df_countries['Country'] = df_countries['Country'].str.strip().str.title()

merged_df = pd.merge(df_leaders, df_countries, on='Country', how='inner')

merged_df.to_csv('merged_leader_country_data.csv', index=False)
print("\nMerged data saved to 'merged_leader_country_data.csv'.")

unmatched_leaders = set(df_leaders['Country']) - set(merged_df['Country'])
unmatched_countries = set(df_countries['Country']) - set(merged_df['Country'])

print("\nUnmatched Leaders (countries in df_leaders not in df_countries):")
print(unmatched_leaders)

print("\nUnmatched Countries (countries in df_countries not in df_leaders):")
print(unmatched_countries)
