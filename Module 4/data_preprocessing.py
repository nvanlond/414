import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv('/Users/niels/code/414/module 4/amazon.csv')

# Keep only relevant columns
data = data[['product_id', 'actual_price', 'rating', 'category']]

# Remove commas and convert numeric columns to floats
numeric_columns = ['actual_price', 'rating']  # Add other numeric columns if needed
for col in numeric_columns:
    data[col] = data[col].replace(',', '', regex=True).astype(float)

# Simplify the category column to the first identifier
data['category'] = data['category'].str.split('|').str[0]

# Apply one-hot encoding to the 'category' column
data = pd.get_dummies(data, columns=['category'], prefix='category')

# Drop rows with missing values
data = data.dropna()

# Save the cleaned and processed data
data.to_csv('processed_data.csv', index=False)
print("Data preparation complete with one-hot encoding. Processed data saved as 'processed_data.csv'.")
