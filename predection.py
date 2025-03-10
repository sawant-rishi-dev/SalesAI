import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

train_data = pd.read_csv('train.csv', parse_dates=['date'])
test_data = pd.read_csv('sample_test.csv', parse_dates=['date'])
oil_data = pd.read_csv('oil.csv', parse_dates=['date'])

train_data = train_data.merge(oil_data, on='date', how='left')
train_data['dcoilwtico'] = train_data['dcoilwtico'].ffill()
train_data['month'] = train_data['date'].dt.month
train_data['day'] = train_data['date'].dt.day
train_data['weekday'] = train_data['date'].dt.weekday
sales_99th_percentile = np.percentile(train_data['sales'], 99)
train_data['sales'] = train_data['sales'].clip(upper=sales_99th_percentile)
train_data['log_sales'] = np.log1p(train_data['sales'])
for lag in range(1, 8):
    train_data[f'sales_lag_{lag}'] = train_data.groupby(['store_nbr', 'family'])['sales'].shift(lag)

train_data['rolling_avg_7'] = train_data.groupby(['store_nbr', 'family'])['sales'].transform(
    lambda x: x.shift(1).rolling(window=7).mean()
)
train_data.fillna(0, inplace=True)
features = ['month', 'day', 'weekday', 'onpromotion', 'dcoilwtico', 'rolling_avg_7'] + [f'sales_lag_{lag}' for lag in range(1, 8)]
X = train_data[features]
y = train_data['log_sales']
split_date = '2023-12-31'
X_train = X[train_data['date'] < split_date]
y_train = y[train_data['date'] < split_date]
X_val = X[train_data['date'] >= split_date]
y_val = y[train_data['date'] >= split_date]
model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
log_y_val_pred = model.predict(X_val)
log_rmse = np.sqrt(mean_squared_error(y_val, log_y_val_pred))
print(f"Validation Log RMSE: {log_rmse}")
required_dates = pd.date_range(start=test_data['date'].min(), end=test_data['date'].max())
oil_data.set_index('date', inplace=True)
oil_data = oil_data.reindex(required_dates)
oil_data['dcoilwtico'] = oil_data['dcoilwtico'].ffill()
oil_data.reset_index(inplace=True)
oil_data.rename(columns={'index': 'date'}, inplace=True)
test_data = test_data.merge(oil_data, on='date', how='left')
last_oil_value = train_data['dcoilwtico'].iloc[-1]
test_data['dcoilwtico'] = test_data['dcoilwtico'].fillna(last_oil_value)
test_data['month'] = test_data['date'].dt.month
test_data['day'] = test_data['date'].dt.day
test_data['weekday'] = test_data['date'].dt.weekday
for lag in range(1, 8):
    last_sales = train_data.groupby(['store_nbr', 'family'])['sales'].last().to_dict()

    test_data[f'sales_lag_{lag}'] = test_data.apply(
        lambda row: last_sales.get((row['store_nbr'], row['family']), 0), axis=1
    )
test_data['rolling_avg_7'] = test_data.groupby(['store_nbr', 'family'])['sales_lag_1'].transform(
    lambda x: x.rolling(window=7).mean()
)
test_data['rolling_avg_7'] = test_data['rolling_avg_7'].ffill()
test_data.fillna(0, inplace=True)
X_test = test_data[features]
log_sales_pred = model.predict(X_test)
test_data['sales'] = np.expm1(log_sales_pred) 
test_data['sales'] = test_data['sales'].clip(lower=0) 
final = test_data[['id']]
final['sales'] = test_data['sales']
final.to_csv('final.csv', index=False)

print("Final file created: 'final.csv'")
uploaded_file = 'train.csv'
train_data_xls = pd.read_csv(uploaded_file, parse_dates=['date'])
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
