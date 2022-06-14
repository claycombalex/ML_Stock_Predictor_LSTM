import simfin as sf
from simfin.names import *
import matplotlib.pyplot as plt
import pandas as pd

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

# Load the datasets to disk
# Income, balance, and cashflow are resampled to fill in NaN values
income_vals = income.loc[ticker, [SHARES_BASIC, REVENUE, GROSS_PROFIT, NET_INCOME]].resample('D').interpolate(method='linear')
balance_vals = balance.loc[ticker, [CASH_EQUIV_ST_INVEST, ACC_NOTES_RECV, PROP_PLANT_EQUIP_NET, TOTAL_LIAB_EQUITY]].resample('D').interpolate(method='linear')
cashflow_vals = cashflow.loc[ticker, [NET_CASH_OPS, NET_CASH_INV, NET_CASH_FIN]].resample('D').interpolate(method='linear')
prices_vals = prices.loc[ticker, [SHARE_PRICE_CLOSE, SHARE_VOLUME]]

# Merge datasets with key "Report Date"
merged_reports = pd.DataFrame.merge(income_vals, balance_vals, how='outer', on='Report Date')
merged_reports = pd.DataFrame.merge(merged_reports, cashflow_vals, how='outer', on='Report Date')
merged_reports.index.name = 'Date'  # Share Price dataset has column title "Date" instead of "Report Date"
merged_reports = pd.DataFrame.merge(merged_reports, prices_vals, how='outer', on='Date')

# Drop entries of the table that have NaN values
merged_reports = merged_reports.dropna()

print(merged_reports.to_string())
