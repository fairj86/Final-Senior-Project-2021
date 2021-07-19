import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import seaborn as sns
import os
import math

from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense,Dropout,LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

plt.rcParams.update({'font.size': 10})
plt.style.use("fivethirtyeight")
sns.set_style('whitegrid')


#Load Data
company = 'HD'
name = 'Home Depot Inc.'

end = datetime.now()
start = datetime(end.year - 5, end.month, end.day)

data= web.DataReader(company, 'yahoo', start, end)

--------------------------------------------------------------------------------

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
test_stationarity(data['Close'])
plt.show()

--------------------------------------------------------------------------------

#Moving Avg
from pylab import rcParams

plt.figure()
rcParams['figure.figsize'] = 10, 6
df_log = np.log(data['Close'])
moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()
plt.legend(loc='best')
plt.title(f'Moving Average for {name}', fontsize=14)
plt.plot(std_dev, color ="black", label = "Standard Deviation", linewidth=1)
plt.plot(moving_avg, color="red", label = "Mean", linewidth=1)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.legend(fontsize=12)
plt.show()

--------------------------------------------------------------------------------

#Arima Model
df_log = np.log(data['Close'])
plt.rcParams.update({'font.size': 10})

#split data into train and training set
train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
plt.figure(figsize=(8,4))
plt.grid(True)
plt.title(f'Closing Test and Train Data for {name}', fontsize= 14)
plt.xlabel('Dates', fontsize=12)
plt.ylabel('Closing Prices', fontsize=12)
plt.plot(df_log, 'green', label='Train data', linewidth=1.5)
plt.plot(test_data, 'blue', label='Test data', linewidth=1.5)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.legend()
plt.tight_layout()
plt.show()

#Programing the ARIMA Model
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

model = ARIMA(train_data, order=(1, 0, 2))  
fitted = model.fit(disp=-1)  
print(fitted.summary())

--------------------------------------------------------------------------------

# Forecast 
df_log = np.log(data['Close'])
plt.rcParams.update({'font.size': 12})
train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]

# Forecast
fc, se, conf = fitted.forecast(3, alpha=0.05)  # 95% confidence
fc_series = pd.Series(fc, index=test_data.index)
lower_series = pd.Series(conf[:, 0], index=test_data.index)
upper_series = pd.Series(conf[:, 1], index=test_data.index)
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train_data, label='Training', linewidth=1)
plt.plot(test_data, color = 'blue', label='Actual Stock Price', linewidth=1)
plt.plot(fc_series, color = 'orange',label='Predicted Stock Price', linewidth=1)
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.10)
plt.title(f'{name} Stock Price Prediction')
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
##LSTM

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

def LSTM_model():
    model = Sequential()
    
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    return model


#Training the Data
model = LSTM_model()
model.summary()
model.compile(optimizer='adam', 
              loss='mean_squared_error')

# Define callbacks

# Save weights only for best model
checkpointer = ModelCheckpoint(filepath = 'weights_best.hdf5', 
                               verbose = 2, 
                               save_best_only = True)

model.fit(x_train, 
          y_train, 
          epochs=25, 
          batch_size = 32,
          callbacks = [checkpointer])


#Testing the Data Accurancy
test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()

test_data = web.DataReader(company, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)


# Predictions on Test Data

x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] ,1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)


#Plot the Test Predictions
plt.figure(figsize=(8, 5))
plt.plot(actual_prices, color="black", label="Actual Price", linewidth=1)
plt.plot(predicted_prices, color="green", label="Predicted Price", linewidth=1)
plt.title(f"{name} Share Price")
plt.xlabel('Time')
plt.ylabel('Share Price')
plt.legend()
plt.tight_layout()
plt.show()



