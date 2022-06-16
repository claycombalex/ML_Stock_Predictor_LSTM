from math import sqrt
import simfin as sf
from simfin.names import *
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.preprocessing as sk
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras as ks
from numpy import concatenate

sf.set_api_key('nmBmGj2WlnAlOBAO6JzosLzzAOgx7aON')

sf.set_data_dir('./simfin_data/')

# Get datasets from simfin.com
# If this step fails, the datasets can be manually downloaded from https://simfin.com/data/bulk
companies = sf.load_companies(index=TICKER, market='us')
income = sf.load_income(variant='TTM', market='us')
balance = sf.load_balance(variant='TTM', market='us')
cashflow = sf.load_cashflow(variant='TTM', market='us')
prices = sf.load_shareprices(market='us', variant = 'daily')

# User inputs the desired stock ticker
ticker = input("Enter stock ticker to predict value: ")
ticker.upper()

# Check if the ticker is part of the SimFin database, if not exit program
try:
    companies.loc[ticker]
except Exception as e:
    print("ERROR: Stock ticker not found in database")
    exit()

# TO-DO add condition for banks and insurance companies because they use different data columns

# Load the relevant parts of the dataset to disk
# Income, balance, and cashflow are resampled to fill in NaN values
income_vals = income.loc[ticker, [SHARES_BASIC, REVENUE, GROSS_PROFIT, NET_INCOME]].resample('D').interpolate(method='linear')
balance_vals = balance.loc[ticker, [CASH_EQUIV_ST_INVEST, ACC_NOTES_RECV, PROP_PLANT_EQUIP_NET, TOTAL_LIAB_EQUITY]].resample('D').interpolate(method='linear')
cashflow_vals = cashflow.loc[ticker, [NET_CASH_OPS, NET_CASH_INV, NET_CASH_FIN]].resample('D').interpolate(method='linear')
prices_vals = prices.loc[ticker, [SHARE_VOLUME, SHARE_PRICE_CLOSE]]

# Merge datasets with date as the key
merged_reports = pd.DataFrame.merge(income_vals, balance_vals, how='outer', on='Report Date')
merged_reports = pd.DataFrame.merge(merged_reports, cashflow_vals, how='outer', on='Report Date')
merged_reports.index.name = 'Date'
merged_reports = pd.DataFrame.merge(merged_reports, prices_vals, how='outer', on='Date')

# Drop entries of the table that have NaN values
merged_reports = merged_reports.dropna()

# Get maximum stock price
maxval = max(merged_reports['Close'].tolist())

# Encode integers
values = merged_reports.values
encoder = sk.LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
values = values.astype('float32')

# Normalize data
scaler = sk.MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(values)

# Convert the scaled values into a supervised learning set
n_vars = 13
supervised_data = pd.DataFrame(scaled_values)
names = list()
names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
supervised_data.columns = names

# Split the data into train and test sets
# Train on the first 1000 days, test on the last 196 days
values = supervised_data.values
train = values[:1100, :]
test = values[1100:, :]

# Split into inputs and outputs
train_X, train_Y = train[:, :-1], train[:, -1]
test_X, test_Y = test[:, :-1], test[:, -1]

# Reshape input to be 3D
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# Create network 
model = ks.models.Sequential()
model.add(ks.layers.LSTM(75, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(ks.layers.Dense(1))
model.compile(loss='mae', optimizer='adam')

# Fit network
history = model.fit(train_X, train_Y, epochs=50, batch_size=72, validation_data=(test_X, test_Y), verbose=2, shuffle=False)

# Plot the history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# Make stock price predictions
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# Calculate RMSE
rmse = sqrt(mean_squared_error(test_Y, yhat))
print(rmse)
print(test_Y * maxval)
print(yhat * maxval)
