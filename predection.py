# Import necessary libraries
6import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Load datasets (make sure to update the file paths)
train_data = pd.read_csv('Z:/Rishi other project/mohit/Sales prediction/train.csv', parse_dates=['date'])
test_data = pd.read_csv('Z:/Rishi other project/mohit/Sales prediction/sample_test.csv', parse_dates=['date'])
oil_data = pd.read_csv('Z:/Rishi other project/mohit/Sales prediction/oil.csv', parse_dates=['date'])
"""sample_submission = pd.read_csv('Z:/Rishi other project/mohit/Sales prediction/sample_submission.csv')"""

# Merge oil data into train_data
train_data = train_data.merge(oil_data, on='date', how='left')

# Forward-fill missing oil prices in train_data
train_data['dcoilwtico'] = train_data['dcoilwtico'].ffill()

# Feature engineering: Time-based features for train_data
train_data['month'] = train_data['date'].dt.month
train_data['day'] = train_data['date'].dt.day
train_data['weekday'] = train_data['date'].dt.weekday

# Handling outliers in sales
sales_99th_percentile = np.percentile(train_data['sales'], 99)
train_data['sales'] = train_data['sales'].clip(upper=sales_99th_percentile)

# Apply log transformation to sales in train_data only, adding 1 to avoid log(0)
train_data['log_sales'] = np.log1p(train_data['sales'])

# Create lag and rolling average features for train_data
for lag in range(1, 8):  # Lag features for the past 7 days
    train_data[f'sales_lag_{lag}'] = train_data.groupby(['store_nbr', 'family'])['sales'].shift(lag)

train_data['rolling_avg_7'] = train_data.groupby(['store_nbr', 'family'])['sales'].transform(
    lambda x: x.shift(1).rolling(window=7).mean()
)

# Fill missing values for lag/rolling features in train_data
train_data.fillna(0, inplace=True)

# Extract features and target variable for train_data
features = ['month', 'day', 'weekday', 'onpromotion', 'dcoilwtico', 'rolling_avg_7'] + [f'sales_lag_{lag}' for lag in range(1, 8)]
X = train_data[features]
y = train_data['log_sales']  # Using log-transformed sales

# Train-test split
split_date = '2023-12-31'  # Adjust split date as per the dataset
X_train = X[train_data['date'] < split_date]
y_train = y[train_data['date'] < split_date]
X_val = X[train_data['date'] >= split_date]
y_val = y[train_data['date'] >= split_date]

# Train XGBoost model
model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Validate model
log_y_val_pred = model.predict(X_val)
log_rmse = np.sqrt(mean_squared_error(y_val, log_y_val_pred))
print(f"Validation Log RMSE: {log_rmse}")

# Handle test data
# Ensure oil data includes all required dates in the test range
required_dates = pd.date_range(start=test_data['date'].min(), end=test_data['date'].max())
oil_data.set_index('date', inplace=True)
oil_data = oil_data.reindex(required_dates)
oil_data['dcoilwtico'] = oil_data['dcoilwtico'].ffill()
oil_data.reset_index(inplace=True)
oil_data.rename(columns={'index': 'date'}, inplace=True)

# Merge oil data into test_data
test_data = test_data.merge(oil_data, on='date', how='left')

# Use the last available oil price from train_data *only for remaining missing values*
last_oil_value = train_data['dcoilwtico'].iloc[-1]
test_data['dcoilwtico'] = test_data['dcoilwtico'].fillna(last_oil_value)

# Add time-based features to test_data
test_data['month'] = test_data['date'].dt.month
test_data['day'] = test_data['date'].dt.day
test_data['weekday'] = test_data['date'].dt.weekday

# For lag features in test_data, use the last available sales values for each (store_nbr, family) from train_data
for lag in range(1, 8):
    # Create a dictionary to map the last sales value for each (store_nbr, family)
    last_sales = train_data.groupby(['store_nbr', 'family'])['sales'].last().to_dict()

    # Apply this last sales value to the corresponding rows in test_data
    test_data[f'sales_lag_{lag}'] = test_data.apply(
        lambda row: last_sales.get((row['store_nbr'], row['family']), 0), axis=1
    )

# Calculate rolling averages based on train_data's sales
test_data['rolling_avg_7'] = test_data.groupby(['store_nbr', 'family'])['sales_lag_1'].transform(
    lambda x: x.rolling(window=7).mean()
)

# Forward-fill rolling averages for test_data
test_data['rolling_avg_7'] = test_data['rolling_avg_7'].ffill()

# Fill remaining missing values
test_data.fillna(0, inplace=True)

# Extract test features
X_test = test_data[features]

# Make predictions on test_data
log_sales_pred = model.predict(X_test)

# Reverse the log transformation for predicted sales (log -> original scale)
test_data['sales'] = np.expm1(log_sales_pred)  # Inverse of log1p
test_data['sales'] = test_data['sales'].clip(lower=0)  # Ensure no negative sales

# Prepare submission file
submission = test_data[['id']]
submission['sales'] = test_data['sales']
submission.to_csv('Z:/Rishi other project/mohit/Sales prediction/Hackerton_Hackers_viva_submission.csv', index=False)

print("Submission file created: 'submission.csv'")

# If you want to upload an Excel sheet for training the data, here's how to read it:
uploaded_file = 'Z:/Rishi other project/mohit/Sales prediction/train.csv'
train_data_xls = pd.read_csv(uploaded_file, parse_dates=['date'])

# The rest of the process is the same as above
train_data_xls = train_data_xls.merge(oil_data, on='date', how='left')
train_data_xls['dcoilwtico'] = train_data_xls['dcoilwtico'].ffill()
train_data_xls['month'] = train_data_xls['date'].dt.month
train_data_xls['day'] = train_data_xls['date'].dt.day
train_data_xls['weekday'] = train_data_xls['date'].dt.weekday
sales_99th_percentile_xls = np.percentile(train_data_xls['sales'], 99)
train_data_xls['sales'] = train_data_xls['sales'].clip(upper=sales_99th_percentile_xls)
train_data_xls['log_sales'] = np.log1p(train_data_xls['sales'])

for lag in range(1, 8):
    train_data_xls[f'sales_lag_{lag}'] = train_data_xls.groupby(['store_nbr', 'family'])['sales'].shift(lag)

train_data_xls['rolling_avg_7'] = train_data_xls.groupby(['store_nbr', 'family'])['sales'].transform(
    lambda x: x.shift(1).rolling(window=7).mean()
)

train_data_xls.fillna(0, inplace=True)

X_xls = train_data_xls[features]
y_xls = train_data_xls['log_sales']

X_train_xls = X_xls[train_data_xls['date'] < split_date]
y_train_xls = y_xls[train_data_xls['date'] < split_date]
X_val_xls = X_xls[train_data_xls['date'] >= split_date]
y_val_xls = y_xls[train_data_xls['date'] >= split_date]

model_xls = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
model_xls.fit(X_train_xls, y_train_xls)

log_y_val_pred_xls = model_xls.predict(X_val_xls)
log_rmse_xls = np.sqrt(mean_squared_error(y_val_xls, log_y_val_pred_xls))
print(f"Validation Log RMSE for Excel data: {log_rmse_xls}")
