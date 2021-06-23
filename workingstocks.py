# Source
import yfinance as yf

#viz
import plotly.graph_objs as go

Apple2 = yf.Ticker("AAPL")
#1 minute history data about Apple
Apple2_historical = Apple2.history(start="2021-04-22", end="2021-04-23", interval="1m")
Apple2_historical

#This example is pulling 3 different tickers organizing by Open/High/Low/Close
TripleThreat = yf.download("AMZN AAPL GOOG", start="2017-01-01", end="2017-04-30")
TripleThreat

#This example is pulling by grouping by ticker
Ticker = yf.download("AMZN AAPL GOOG", start="2017-01-01",
                    end="2017-04-30", group_by='tickers')
Ticker

#price to earning ratio
aapl = yf.Ticker("aapl")
aapl.info['forwardPE']

#dividends
aapl.info['dividendRate']

#or this way[dividends]
aapl.dividends

#marketcap
aapl.info["marketCap"]

#current volume
aapl.info["volume"]

#average volume over the last 24 hours
aapl.info["averageVolume"]

#average volume in the last 10 days
aapl.info["averageVolume10days"]

import pandas as pd

tickers_list = ["aapl", "goog", "amzn", "BAC", "BA"] # example list
tickers_data= {} # empty dictionary

for ticker in tickers_list:
    ticker_object = yf.Ticker(ticker)

    #convert info() output from dictionary to dataframe
    temp = pd.DataFrame.from_dict(ticker_object.info, orient="index")
    temp.reset_index(inplace=True)
    temp.columns = ["Attribute", "Recent"]
    
    # add (ticker, dataframe) to main dictionary
    tickers_data[ticker] = temp

tickers_data
