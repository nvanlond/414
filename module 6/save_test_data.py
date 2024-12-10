import pandas as pd

# Load the dataset
df = pd.read_csv("/Users/niels/code/414/module 6/Airbnb Data/Listings.csv")

# Separate rows into testing and training sets using a for loop
testing_set = []
training_set = []

for index, row in df.iterrows():
    if index % 5 == 0:  # Every 5th row
        testing_set.append(row)
    else:
        training_set.append(row)

# Convert the lists back to DataFrames
testing_set = pd.DataFrame(testing_set, columns=df.columns)
training_set = pd.DataFrame(training_set, columns=df.columns)

# Save the testing set to a CSV file
testing_set.to_csv("price_predict_testing.csv", index=False)

# Save the remaining data as the training set
training_set.to_csv("price_predict_training.csv", index=False)

print("Testing set saved as 'price_predict_testing.csv'")
print("Remaining training set saved as 'price_predict_training.csv'")
