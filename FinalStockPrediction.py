import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
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

--------------------------------------------------------------------------------
#Mean and Standard Deviation- AAPL
plt.figure()
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    #Plot rolling statistics:
    plt.plot(timeseries, color='yellow',label='Original', linewidth=1.5)
    plt.plot(rolmean, color='red', label='Rolling Mean', linewidth=1.5)
    plt.plot(rolstd, color='black', label = 'Rolling Std', linewidth=1.5)
    plt.legend(loc='best')
    plt.title("Rolling Mean and Standard Deviation for Apple Inc.")
    plt.show(block=False)
    
    print("Results of dickey fuller test")
    adft = adfuller(timeseries,autolag='AIC')
    # output for dft will give us without defining what the values are.
    #hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    print(output)
    
plt.figure()
test_stationarity(AAPL['Close'])
plt.show()

--------------------------------------------------------------------------------

#Mean and Standard Deviation- AMZN
plt.figure()
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    #Plot rolling statistics:
    plt.plot(timeseries, color='yellow',label='Original', linewidth=1.5)
    plt.plot(rolmean, color='red', label='Rolling Mean', linewidth=1.5)
    plt.plot(rolstd, color='black', label = 'Rolling Std', linewidth=1.5)
    plt.legend(loc='best')
    plt.title("Rolling Mean and Standard Deviation for Amazon.com,Inc.")
    plt.show(block=False)
    
    print("Results of dickey fuller test")
    adft = adfuller(timeseries,autolag='AIC')
    # output for dft will give us without defining what the values are.
    #hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    print(output)

plt.figure()
test_stationarity(AMZN['Close'])
plt.show()

--------------------------------------------------------------------------------

#Mean and Standard Deviation- DIS
plt.figure()
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    #Plot rolling statistics:
    plt.plot(timeseries, color='yellow',label='Original', linewidth=1.5)
    plt.plot(rolmean, color='red', label='Rolling Mean', linewidth=1.5)
    plt.plot(rolstd, color='black', label = 'Rolling Std', linewidth=1.5)
    plt.legend(loc='best')
    plt.title("Rolling Mean and Standard Deviation for Disney")
    plt.show(block=False)
    
    print("Results of dickey fuller test")
    adft = adfuller(timeseries,autolag='AIC')
    # output for dft will give us without defining what the values are.
    #hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    print(output)
plt.figure()
test_stationarity(DIS['Close'])
plt.show()

--------------------------------------------------------------------------------

#Mean and Standard Deviation- GOOG
plt.figure()
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    #Plot rolling statistics:
    plt.plot(timeseries, color='yellow',label='Original', linewidth=1.5)
    plt.plot(rolmean, color='red', label='Rolling Mean', linewidth=1.5)
    plt.plot(rolstd, color='black', label = 'Rolling Std', linewidth=1.5)
    plt.legend(loc='best')
    plt.title("Rolling Mean and Standard Deviation for Google Inc")
    plt.show(block=False)
    
    print("Results of dickey fuller test")
    adft = adfuller(timeseries,autolag='AIC')
    # output for dft will give us without defining what the values are.
    #hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    print(output)
plt.figure()
test_stationarity(GOOG['Close'])
plt.show()

--------------------------------------------------------------------------------

#Mean and Standard Deviation- HD
plt.figure()
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    #Plot rolling statistics:
    plt.plot(timeseries, color='yellow',label='Original', linewidth=1.5)
    plt.plot(rolmean, color='red', label='Rolling Mean', linewidth=1.5)
    plt.plot(rolstd, color='black', label = 'Rolling Std', linewidth=1.5)
    plt.legend(loc='best')
    plt.title("Rolling Mean and Standard Deviation for Home Depot, Inc.")
    plt.show(block=False)
    
    print("Results of dickey fuller test")
    adft = adfuller(timeseries,autolag='AIC')
    # output for dft will give us without defining what the values are.
    #hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    print(output)
plt.figure()
test_stationarity(HD['Close'])
plt.show()

--------------------------------------------------------------------------------

#Mean and Standard Deviation- WMT
plt.figure()
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    #Plot rolling statistics:
    plt.plot(timeseries, color='yellow',label='Original', linewidth=1.5)
    plt.plot(rolmean, color='red', label='Rolling Mean', linewidth=1.5)
    plt.plot(rolstd, color='black', label = 'Rolling Std', linewidth=1.5)
    plt.legend(loc='best')
    plt.title("Rolling Mean and Standard Deviation for Walmart Inc.")
    plt.show(block=False)
    
    print("Results of dickey fuller test")
    adft = adfuller(timeseries,autolag='AIC')
    # output for dft will give us without defining what the values are.
    #hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    print(output)
plt.figure()
test_stationarity(WMT['Close'])
plt.show()

--------------------------------------------------------------------------------
from pylab import rcParams

plt.figure()
rcParams['figure.figsize'] = 10, 6
df_log = np.log(AAPL['Close'])
moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()
plt.legend(loc='best')
plt.title('Moving Average for Apple Inc.')
plt.plot(std_dev, color ="black", label = "Standard Deviation", linewidth=1)
plt.plot(moving_avg, color="red", label = "Mean", linewidth=1)
plt.legend()
plt.show()
--------------------------------------------------------------------------------
plt.figure()
rcParams['figure.figsize'] = 10, 6
df_log = np.log(AMZN['Close'])
moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()
plt.legend(loc='best')
plt.title('Moving Average for Amazon.com, Inc.')
plt.plot(std_dev, color ="black", label = "Standard Deviation", linewidth=1)
plt.plot(moving_avg, color="red", label = "Mean", linewidth=1)
plt.legend()
plt.show()
--------------------------------------------------------------------------------
plt.figure()
rcParams['figure.figsize'] = 10, 6
df_log = np.log(DIS['Close'])
moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()
plt.legend(loc='best')
plt.title('Moving Average for Disney')
plt.plot(std_dev, color ="black", label = "Standard Deviation", linewidth=1)
plt.plot(moving_avg, color="red", label = "Mean", linewidth=1)
plt.legend()
plt.show()
--------------------------------------------------------------------------------
plt.figure()
rcParams['figure.figsize'] = 10, 6
df_log = np.log(GOOG['Close'])
moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()
plt.legend(loc='best')
plt.title('Moving Average for Google Inc.')
plt.plot(std_dev, color ="black", label = "Standard Deviation", linewidth=1)
plt.plot(moving_avg, color="red", label = "Mean", linewidth=1)
plt.legend()
plt.show()
--------------------------------------------------------------------------------
plt.figure()
rcParams['figure.figsize'] = 10, 6
df_log = np.log(HD['Close'])
moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()
plt.legend(loc='best')
plt.title('Moving Average for Home Depot Inc.')
plt.plot(std_dev, color ="black", label = "Standard Deviation", linewidth=1)
plt.plot(moving_avg, color="red", label = "Mean", linewidth=1)
plt.legend()
plt.show()
--------------------------------------------------------------------------------
plt.figure()
rcParams['figure.figsize'] = 10, 6
df_log = np.log(WMT['Close'])
moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()
plt.legend(loc='best')
plt.title('Moving Average for Walmart Inc.')
plt.plot(std_dev, color ="black", label = "Standard Deviation", linewidth=1)
plt.plot(moving_avg, color="red", label = "Mean", linewidth=1)
plt.legend()
plt.show()

--------------------------------------------------------------------------------
#Arima Model- AAPL
df_log = np.log(AAPL['Close'])
plt.rcParams.update({'font.size': 10})

#split data into train and training set
train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
plt.figure(figsize=(10,5))
plt.grid(True)
plt.title('Closing Test and Train Data for Apple Inc.')
plt.xlabel('Dates', fontsize=12)
plt.ylabel('Closing Prices', fontsize=12)
plt.plot(df_log, 'green', label='Train data', linewidth=1.5)
plt.plot(test_data, 'blue', label='Test data', linewidth=1.5)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.legend()
plt.show()

model = ARIMA(train_data, order=(3, 1, 2))  
fitted = model.fit(disp=-1)  
print(fitted.summary())

model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
test='adf',       # use adftest to find optimal 'd'
max_p=3, max_q=3, # maximum p and q
m=1,              # frequency of series
d=None,           # let model determine 'd'
seasonal=False,   # No Seasonality
start_P=0, 
D=0, 
trace=True,
error_action='ignore',  
suppress_warnings=True, 
stepwise=True,)
print(model_autoARIMA.summary())

plt.figure()
model_autoARIMA.plot_diagnostics()
plt.subplots_adjust(top=1.4,bottom=1.25, left=1.25)
plt.tight_layout()
plt.show()


model = ARIMA(train_data, order=(3, 1, 2))
fitted = model.fit(disp=-1)
print(fitted.summary())

--------------------------------------------------------------------------------
#Arima Model- AMZN
df_log = np.log(AMZN['Close'])
plt.rcParams.update({'font.size': 10})

#split data into train and training set
train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
plt.figure(figsize=(10,5))
plt.grid(True)
plt.title('Closing Test and Train Data for Amazon.com, Inc.')
plt.xlabel('Dates', fontsize=12)
plt.ylabel('Closing Prices', fontsize=12)
plt.plot(df_log, 'green', label='Train data', linewidth=1.5)
plt.plot(test_data, 'blue', label='Test data', linewidth=1.5)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.legend()
plt.show()

model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
test='adf',       # use adftest to find optimal 'd'
max_p=3, max_q=3, # maximum p and q
m=1,              # frequency of series
d=None,           # let model determine 'd'
seasonal=False,   # No Seasonality
start_P=0, 
D=0, 
trace=True,
error_action='ignore',  
suppress_warnings=True, 
stepwise=True)
print(model_autoARIMA.summary())

plt.figure()
model_autoARIMA.plot_diagnostics()
plt.subplots_adjust(top=1.4,bottom=1.25, left=1.25)
plt.tight_layout()
plt.show()


model = ARIMA(train_data, order=(3, 1, 2))
fitted = model.fit(disp=-1)
print(fitted.summary())

--------------------------------------------------------------------------------
#Arima Model- DIS
df_log = np.log(DIS['Close'])
plt.rcParams.update({'font.size': 10})

#split data into train and training set
train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
plt.figure(figsize=(10,5))
plt.grid(True)
plt.title('Closing Test and Train Data for Disney Inc.')
plt.xlabel('Dates', fontsize=12)
plt.ylabel('Closing Prices', fontsize=12)
plt.plot(df_log, 'green', label='Train data', linewidth=1.5)
plt.plot(test_data, 'blue', label='Test data', linewidth=1.5)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.legend()
plt.show()

model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
test='adf',       # use adftest to find optimal 'd'
max_p=3, max_q=3, # maximum p and q
m=1,              # frequency of series
d=None,           # let model determine 'd'
seasonal=False,   # No Seasonality
start_P=0, 
D=0, 
trace=True,
error_action='ignore',  
suppress_warnings=True, 
stepwise=True)
print(model_autoARIMA.summary())

plt.figure()
model_autoARIMA.plot_diagnostics()
plt.subplots_adjust(top=1.4,bottom=1.25, left=1.25)
plt.tight_layout()
plt.show()


model = ARIMA(train_data, order=(3, 1, 2))
fitted = model.fit(disp=-1)
print(fitted.summary())

--------------------------------------------------------------------------------
#Arima Model- GOOG
df_log = np.log(GOOG['Close'])
plt.rcParams.update({'font.size': 10})

#split data into train and training set
train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
plt.figure(figsize=(10,5))
plt.grid(True)
plt.title('Closing Test and Train Data for Google Inc.')
plt.xlabel('Dates', fontsize=12)
plt.ylabel('Closing Prices', fontsize=12)
plt.plot(df_log, 'green', label='Train data', linewidth=1.5)
plt.plot(test_data, 'blue', label='Test data', linewidth=1.5)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.legend()
plt.show()

model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
test='adf',       # use adftest to find optimal 'd'
max_p=3, max_q=3, # maximum p and q
m=1,              # frequency of series
d=None,           # let model determine 'd'
seasonal=False,   # No Seasonality
start_P=0, 
D=0, 
trace=True,
error_action='ignore',  
suppress_warnings=True, 
stepwise=True)
print(model_autoARIMA.summary())

plt.figure()
model_autoARIMA.plot_diagnostics()
plt.subplots_adjust(top=1.4,bottom=1.25, left=1.25)
plt.tight_layout()
plt.show()


model = ARIMA(train_data, order=(1, 0, 1))
fitted = model.fit()
print(fitted.summary())

--------------------------------------------------------------------------------
#Arima Model- HD
df_log = np.log(HD['Close'])
plt.rcParams.update({'font.size': 10})

#split data into train and training set
train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
plt.figure(figsize=(10,5))
plt.grid(True)
plt.title('Closing Test and Train Data for Home Depot Inc.')
plt.xlabel('Dates', fontsize=12)
plt.ylabel('Closing Prices', fontsize=12)
plt.plot(df_log, 'green', label='Train data', linewidth=1.5)
plt.plot(test_data, 'blue', label='Test data', linewidth=1.5)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.legend()
plt.show()

model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
test='adf',       # use adftest to find optimal 'd'
max_p=3, max_q=3, # maximum p and q
m=1,              # frequency of series
d=None,           # let model determine 'd'
seasonal=False,   # No Seasonality
start_P=0, 
D=0, 
trace=True,
error_action='ignore',  
suppress_warnings=True, 
stepwise=True)
print(model_autoARIMA.summary())

plt.figure()
model_autoARIMA.plot_diagnostics()
plt.subplots_adjust(top=1.4,bottom=1.25, left=1.25)
plt.tight_layout()
plt.show()


model = ARIMA(train_data, order=(3, 1, 2))
fitted = model.fit(disp=-1)
print(fitted.summary())

--------------------------------------------------------------------------------
#Arima Model- WMT
df_log = np.log(WMT['Close'])
plt.rcParams.update({'font.size': 10})

#split data into train and training set
train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
plt.figure(figsize=(10,5))
plt.grid(True)
plt.title('Closing Test and Train Data for Walmart Inc.')
plt.xlabel('Dates', fontsize=12)
plt.ylabel('Closing Prices', fontsize=12)
plt.plot(df_log, 'green', label='Train data', linewidth=1.5)
plt.plot(test_data, 'blue', label='Test data', linewidth=1.5)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.legend()
plt.show()

model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
test='adf',       # use adftest to find optimal 'd'
max_p=3, max_q=3, # maximum p and q
m=1,              # frequency of series
d=None,           # let model determine 'd'
seasonal=False,   # No Seasonality
start_P=0, 
D=0, 
trace=True,
error_action='ignore',  
suppress_warnings=True, 
stepwise=True)
print(model_autoARIMA.summary())

plt.figure()
model_autoARIMA.plot_diagnostics()
plt.subplots_adjust(top=1.4,bottom=1.25, left=1.25)
plt.tight_layout()
plt.show()


model = ARIMA(train_data, order=(3, 1, 2))
fitted = model.fit(disp=-1)
print(fitted.summary())

--------------------------------------------------------------------------------

# Forecast- AAPL
df_log = np.log(AAPL['Close'])
plt.rcParams.update({'font.size': 12})
train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
plt.figure()
fc, se, conf = fitted.forecast(126, alpha=0.05)  # 95% confidence

fc_series = pd.Series(fc, index=test_data.index)
lower_series = pd.Series(conf[:, 0], index=test_data.index)
upper_series = pd.Series(conf[:, 1], index=test_data.index)
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train_data, label='training', linewidth=1)
plt.plot(test_data, color = 'blue', label='Actual Stock Price', linewidth=1)
plt.plot(fc_series, color = 'orange',label='Predicted Stock Price', linewidth=1)
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.10)
plt.title('AAPL Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Actual Stock Price')
plt.legend(loc='upper left', fontsize=10)
plt.show()

# report performance
mse = mean_squared_error(test_data, fc)
print('MSE: '+str(mse))
mae = mean_absolute_error(test_data, fc)
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(test_data, fc))
print('RMSE: '+str(rmse))
mape = np.mean(np.abs(fc - test_data)/np.abs(test_data))
print('MAPE: '+str(mape))

--------------------------------------------------------------------------------

# Forecast- AMZN
df_log = np.log(AMZN['Close'])
plt.rcParams.update({'font.size': 12})
train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
plt.figure()
fc, se, conf = fitted.forecast(126, alpha=0.05)  # 95% confidence

fc_series = pd.Series(fc, index=test_data.index)
lower_series = pd.Series(conf[:, 0], index=test_data.index)
upper_series = pd.Series(conf[:, 1], index=test_data.index)
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train_data, label='training', linewidth=1)
plt.plot(test_data, color = 'blue', label='Actual Stock Price', linewidth=1)
plt.plot(fc_series, color = 'orange',label='Predicted Stock Price', linewidth=1)
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.10)
plt.title('AMZN Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Actual Stock Price')
plt.legend(loc='upper left', fontsize=10)
plt.show()

# report performance
mse = mean_squared_error(test_data, fc)
print('MSE: '+str(mse))
mae = mean_absolute_error(test_data, fc)
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(test_data, fc))
print('RMSE: '+str(rmse))
mape = np.mean(np.abs(fc - test_data)/np.abs(test_data))
print('MAPE: '+str(mape))
--------------------------------------------------------------------------------

# Forecast- DIS
df_log = np.log(DIS['Close'])
train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
plt.rcParams.update({'font.size': 12})
plt.figure()
fc, se, conf = fitted.forecast(126, alpha=0.05)  # 95% confidence

fc_series = pd.Series(fc, index=test_data.index)
lower_series = pd.Series(conf[:, 0], index=test_data.index)
upper_series = pd.Series(conf[:, 1], index=test_data.index)
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train_data, label='training', linewidth=1)
plt.plot(test_data, color = 'blue', label='Actual Stock Price', linewidth=1)
plt.plot(fc_series, color = 'orange',label='Predicted Stock Price', linewidth=1)
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.10)
plt.title('DIS Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Actual Stock Price')
plt.legend(loc='upper left', fontsize=10)
plt.show()

# report performance
mse = mean_squared_error(test_data, fc)
print('MSE: '+str(mse))
mae = mean_absolute_error(test_data, fc)
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(test_data, fc))
print('RMSE: '+str(rmse))
mape = np.mean(np.abs(fc - test_data)/np.abs(test_data))
print('MAPE: '+str(mape))

--------------------------------------------------------------------------------

# Forecast- GOOG
df_log = np.log(GOOG['Close'])
train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
plt.rcParams.update({'font.size': 12})
plt.figure()
fc, se, conf = fitted.forecast(519, alpha=0.05)  # 95% confidence

fc_series = pd.Series(fc, index=test_data.index)
lower_series = pd.Series(conf[:, 0], index=test_data.index)
upper_series = pd.Series(conf[:, 1], index=test_data.index)
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train_data, label='training', linewidth=1)
plt.plot(test_data, color = 'blue', label='Actual Stock Price', linewidth=1)
plt.plot(fc_series, color = 'orange',label='Predicted Stock Price', linewidth=1)
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.10)
plt.title('GOOG Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Actual Stock Price')
plt.legend(loc='upper left', fontsize=10)
plt.show()

# report performance
mse = mean_squared_error(test_data, fc)
print('MSE: '+str(mse))
mae = mean_absolute_error(test_data, fc)
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(test_data, fc))
print('RMSE: '+str(rmse))
mape = np.mean(np.abs(fc - test_data)/np.abs(test_data))
print('MAPE: '+str(mape))

--------------------------------------------------------------------------------

# Forecast- HD
df_log = np.log(HD['Close'])
train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
plt.rcParams.update({'font.size': 12})
plt.figure()
fc, se, conf = fitted.forecast(126, alpha=0.05)  # 95% confidence

fc_series = pd.Series(fc, index=test_data.index)
lower_series = pd.Series(conf[:, 0], index=test_data.index)
upper_series = pd.Series(conf[:, 1], index=test_data.index)
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train_data, label='training', linewidth=1)
plt.plot(test_data, color = 'blue', label='Actual Stock Price', linewidth=1)
plt.plot(fc_series, color = 'orange',label='Predicted Stock Price', linewidth=1)
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.10)
plt.title('HD Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Actual Stock Price')
plt.legend(loc='upper left', fontsize=10)
plt.show()

# report performance
mse = mean_squared_error(test_data, fc)
print('MSE: '+str(mse))
mae = mean_absolute_error(test_data, fc)
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(test_data, fc))
print('RMSE: '+str(rmse))
mape = np.mean(np.abs(fc - test_data)/np.abs(test_data))
print('MAPE: '+str(mape))

--------------------------------------------------------------------------------

# Forecast- WMT
df_log = np.log(WMT['Close'])
train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
plt.rcParams.update({'font.size': 12})
plt.figure()
fc, se, conf = fitted.forecast(126, alpha=0.05)  # 95% confidence

fc_series = pd.Series(fc, index=test_data.index)
lower_series = pd.Series(conf[:, 0], index=test_data.index)
upper_series = pd.Series(conf[:, 1], index=test_data.index)
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train_data, label='training', linewidth=1)
plt.plot(test_data, color = 'blue', label='Actual Stock Price', linewidth=1)
plt.plot(fc_series, color = 'orange',label='Predicted Stock Price', linewidth=1)
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.10)
plt.title('WMT Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Actual Stock Price')
plt.legend(loc='upper left', fontsize=10)
plt.show()

# report performance
mse = mean_squared_error(test_data, fc)
print('MSE: '+str(mse))
mae = mean_absolute_error(test_data, fc)
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(test_data, fc))
print('RMSE: '+str(rmse))
mape = np.mean(np.abs(fc - test_data)/np.abs(test_data))
print('MAPE: '+str(mape))








