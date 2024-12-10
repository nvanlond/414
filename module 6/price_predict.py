import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load and read datasets
train_file = "/Users/niels/code/414/module 6/Airbnb Data/price_predict_training.csv"
test_file = "/Users/niels/code/414/module 6/Airbnb Data/price_predict_testing.csv"

training_data = pd.read_csv(train_file, low_memory=False)
testing_data = pd.read_csv(test_file, low_memory=False)

# Handle price outliers (top 10% of prices)
price_threshold = training_data['price'].quantile(0.9)
training_data = training_data[training_data['price'] < price_threshold]
testing_data = testing_data[testing_data['price'] < price_threshold]

# Parse amenities and count them
training_data['amenities'] = training_data['amenities'].fillna("[]")
testing_data['amenities'] = testing_data['amenities'].fillna("[]")
training_data['amenities_count'] = training_data['amenities'].apply(lambda x: len(eval(x)))
testing_data['amenities_count'] = testing_data['amenities'].apply(lambda x: len(eval(x)))

# Log-transform the target variable
training_data['log_price'] = np.log1p(training_data['price'])
testing_data['log_price'] = np.log1p(testing_data['price'])

# Select features for model training
features = ['bedrooms', 'accommodates', 'review_scores_rating', 'amenities_count', 'neighbourhood', 'district', 'city']
X_train = training_data[features]
y_train = training_data['log_price']
X_test = testing_data[features]
y_test = testing_data['log_price']

# Handle missing values for numeric columns by imputing the mean
numeric_cols = X_train.select_dtypes(include=[np.number]).columns
X_train[numeric_cols] = X_train[numeric_cols].fillna(X_train[numeric_cols].mean())
X_test[numeric_cols] = X_test[numeric_cols].fillna(X_test[numeric_cols].mean())

# Concatenate training and test data to apply one-hot encoding consistently
X = pd.concat([X_train, X_test], axis=0)

# Apply One-Hot Encoding to categorical variables
X = pd.get_dummies(X, columns=['neighbourhood', 'district', 'city'], drop_first=True)

# Split the data back into train and test sets
X_train = X.iloc[:len(X_train)]
X_test = X.iloc[len(X_train):]

# Apply Standard Scaling to the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Ridge Regression model
regressor = Ridge(alpha=1000)
regressor.fit(X_train_scaled, y_train)

# Make predictions
y_pred_log = regressor.predict(X_test_scaled)
y_pred = np.expm1(y_pred_log)  

# Evaluate model performance
rmse = mean_squared_error(testing_data['price'], y_pred) ** 0.5
mae = mean_absolute_error(testing_data['price'], y_pred)
r2 = r2_score(y_test, y_pred_log)

# Display results
print(f"Root Mean Squared Error (Original Scale): {rmse:.2f}")
print(f"Mean Absolute Error (Original Scale): {mae:.2f}")
print(f"R-squared (Log-Scale): {r2:.2f}")

residuals = testing_data['price'] - y_pred

# Sort residuals to find the largest errors (top 5 errors)
wrong_samples = residuals.abs().sort_values(ascending=False).head(5)

top_indices = wrong_samples.index

# Extract listing IDs, actual prices, and predicted prices for these top 5 errors
top_wrong_samples = testing_data.loc[top_indices, ['listing_id', 'price']]  # Listing ID and actual prices
top_wrong_samples['predicted_price'] = y_pred[top_indices]  # Predicted prices

# Print the wrong samples without the index column
print(top_wrong_samples.to_string(index=False))


# Visualization - Predicted vs Actual Price
plt.figure(figsize=(8, 6))
plt.scatter(testing_data['price'], y_pred, alpha=0.5, color='blue', edgecolors='k')
plt.plot([0, max(testing_data['price'])], [0, max(testing_data['price'])], color='red', linestyle='--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted vs Actual Price")
plt.ylim(0, 10000)  # Set y-axis limit to 10,000 for better visualization
plt.show()
