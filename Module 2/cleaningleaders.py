import pandas as pd

# Load the CSV file
file_path = '/Users/niels/code/414/module 2/world data.csv'  
leaders_df = pd.read_csv(file_path)

# Drop the second column (flag) by specifying its index
leaders_df.drop(leaders_df.columns[6], axis=1, inplace=True)
leaders_df.drop(leaders_df.columns[0], axis=1, inplace=True)

# Display the DataFrame to verify the column is removed
print(leaders_df.head())

# If you want to save the updated DataFrame back to a CSV file
leaders_df.to_csv('cleaned_leaders_data.csv', index=False)
