import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
from pandas_datareader import data as pdr
import seaborn as sns
import os
import math
from datetime import datetime
import yfinance as yf

from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,Dropout,LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

plt.rcParams.update({'font.size': 10})
plt.style.use("fivethirtyeight")
sns.set_style('whitegrid')


#Load Data
company = 'GOOG'
name = 'Alphabet.com'
full_data= pd.read_csv('GOOG-1.csv')
data = pd.read_csv('GOOG-2.csv')
#Mean and Standard Deviation
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
    plt.legend(loc='best', fontsize= 12)
    plt.xticks(size = 10)
    plt.yticks(size = 10)
    plt.title(f"Rolling Mean and Standard Deviation for {name}", fontsize=14)
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
test_stationarity(full_data['Close'])
plt.show()



#Moving Avg
from pylab import rcParams

plt.figure()
rcParams['figure.figsize'] = 10, 6
data_log = np.log(full_data['Close'])
moving_avg = data_log.rolling(12).mean()
std_dev = data_log.rolling(12).std()
plt.legend(loc='best')
plt.title(f'Moving Average for {name}', fontsize=14)
plt.plot(std_dev, color ="black", label = "Standard Deviation", linewidth=1)
plt.plot(moving_avg, color="red", label = "Mean", linewidth=1)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.legend(fontsize=12)
plt.show()

--------------------------------------------------------------------------------
#ARIMA

from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
df_log = np.log(data['Close'])

model = pm.auto_arima(df_log, start_p=1, start_q=1,
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

print(model.summary())


# Forecast
n_periods = 50
fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = np.arange(len(df_log), len(df_log)+n_periods)

# make series for plotting purpose
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

plt.rcParams.update({'font.size': 12})
train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]


plt.figure(figsize=(8,5), dpi=100)
plt.plot(train_data, label='Training', linewidth=1)
plt.plot(test_data, color = 'blue', label='Actual Stock Price', linewidth=1)
plt.plot(fc_series, color = 'orange',label='Predicted Stock Price', linewidth=1)
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.10)
plt.title(f'{name} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Actual Stock Price')
plt.legend(loc='upper left', fontsize=10)
plt.tight_layout()
plt.show()


--------------------------------------------------------------------------------
#LSTM

# Convert the dataframe to a numpy array
dataset = df_log.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .95 ))

training_data_len

#Prepare Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


#Build the model

from tensorflow.keras.layers import LSTM

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002 
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
rmse

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

# Visualize the data
plt.figure(figsize=(6,5))
plt.title(f'LSTM Model for {name}')
plt.xlabel('Time', fontsize=14)
plt.ylabel('Close Price USD ($)', fontsize=14)
plt.plot(train['Close'], linewidth=1)
plt.plot(valid[['Close', 'Predictions']], linewidth=1)
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.tight_layout()
plt.show()




