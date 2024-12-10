import pandas as pd

# Load the dataset
df = pd.read_csv("/Users/niels/code/414/module 6/Airbnb Data/Listings.csv")

# Drop the unnecessary columns
columns_to_drop = ['name', 'host_id', 'host_location']
df_cleaned = df.drop(columns=columns_to_drop)

# Save the cleaned dataset to a new file
df_cleaned.to_csv("airbnb_listings_cleaned.csv", index=False)

print("Data cleaned and saved as 'airbnb_listings_cleaned.csv'.")

# Load the datasets
training_data = pd.read_csv("/Users/niels/code/414/module 6/Airbnb Data/price_predict_training.csv", low_memory=False)
testing_data = pd.read_csv("/Users/niels/code/414/module 6/Airbnb Data/price_predict_testing.csv", low_memory=False)

# Remove 'host_response_time' from both datasets
if 'host_response_time' in training_data.columns:
    training_data = training_data.drop(columns=['host_response_time'])
if 'host_response_time' in testing_data.columns:
    testing_data = testing_data.drop(columns=['host_response_time'])

# Save the cleaned datasets
training_data.to_csv("/Users/niels/code/414/module 6/Airbnb Data/price_predict_training_cleaned.csv", index=False)
testing_data.to_csv("/Users/niels/code/414/module 6/Airbnb Data/price_predict_testing_cleaned.csv", index=False)

print("Removed 'host_response_time' and saved cleaned datasets.")

