import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
import math

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# For reading stock data from yahoo
from pandas_datareader.data import DataReader

# For time stamps
from datetime import datetime

# The tech stocks we'll use for this analysis
tech_list = ['AAPL', 'AMZN', 'DIS', 'GOOG', 'HD', 'WMT']

# Set up End and Start times for data grab
end = datetime.now()
start = datetime(end.year - 5, end.month, end.day)


#For loop for grabing yahoo finance data and setting as a dataframe
for stock in tech_list:   
    # Set DataFrame as the Stock Ticker
    globals()[stock] = DataReader(stock, 'yahoo', start, end)
    
# for company, company_name in zip(company_list, tech_list):
#     company["company_name"] = company_name

company_list = [AAPL, AMZN, DIS, GOOG, HD, WMT]
company_name = ["APPLE", "AMAZON", "DISNEY", "GOOGLE", "HOMEDEPOT", "WALMART"]

for company, com_name in zip(company_list, company_name):
    company["company_name"] = com_name
    
df = pd.concat(company_list, axis=0)
df.tail(10)

# Summary Stats
AAPL.describe()

# General info
AAPL.info()

# Let's see a historical view of the closing price

--------------------------------------------------------------------------------
plt.figure(figsize=(9,5))
plt.subplots_adjust(top=1.25,bottom=1.4)

for i, company in enumerate(company_list, 1):
    plt.subplot(3, 2, i)
    company['Adj Close'].plot(linewidth=1)
    plt.ylabel('Adj Close',fontsize = 12)
    plt.xlabel(None)
    plt.title(f"Closing Price of {tech_list[i - 1]}", fontsize= 14)
    plt.xticks(size = 10)
    plt.yticks(size = 10)
    
plt.tight_layout()
plt.show()

--------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, company in enumerate(company_list, 1):
    plt.subplot(3, 2, i)
    company['Volume'].plot(linewidth=1)
    plt.ylabel('Volume', fontsize = 10)
    plt.xlabel(None)
    plt.title(f"Sales Volume for {tech_list[i - 1]}", fontsize = 12)
    plt.xticks(size = 10)
    plt.yticks(size = 9)
    
    
plt.tight_layout()
plt.show()

