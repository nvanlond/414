import pandas as pd

# Load the dataset
file_path = '/Users/niels/code/414/maps/U.S._Life_Expectancy_at_Birth_by_State_and_Census_Tract_-_2010-2015.csv'
df = pd.read_csv(file_path)

# Filter for Maryland data
maryland_df = df[df['State'].str.contains('Maryland', na=False)]

# Select only the required columns
maryland_df = maryland_df[['County', 'Census Tract Number', 'Life Expectancy']]

# Drop rows with missing values in important columns
maryland_df = maryland_df.dropna(subset=['Census Tract Number', 'Life Expectancy'])

# View the cleaned data
print(maryland_df.head())

# Save the cleaned data to a new CSV file
maryland_df.to_csv('cleaned_maryland_life_expectancy.csv', index=False)




