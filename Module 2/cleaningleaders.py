import pandas as pd

# Load the CSV file
file_path = '/Users/niels/code/414/module 2/world data.csv'  
leaders_df = pd.read_csv(file_path)

# Drop the second column (flag)
leaders_df.drop(leaders_df.columns[6], axis=1, inplace=True)
leaders_df.drop(leaders_df.columns[0], axis=1, inplace=True)

print(leaders_df.head())

leaders_df.to_csv('cleaned_leaders_data.csv', index=False)
