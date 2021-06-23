# Raw Package
import numpy as np
import pandas as pd

#Apple Source
import yfinance as yf

#Apple viz
import plotly.graph_objs as go

#Interval required 1 minute
#Apple
Apple = yf.download(tickers='AAPL', period='1d', interval='1m')


#declare figure
fig = go.Figure()

#Candlestick
fig.add_trace(go.Candlestick(x=Apple.index,
                open=Apple['Open'],
                high=Apple['High'],
                low=Apple['Low'],
                close=Apple['Close'], name = 'market Apple'))

# Add titles
fig.update_layout(
    title='Apple live share price evolution',
    yaxis_title='Stock Price (USD per Shares)')

# X-Axes
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=15, label="15m", step="minute", stepmode="backward"),
            dict(count=45, label="45m", step="minute", stepmode="backward"),
            dict(count=1, label="HTD", step="hour", stepmode="todate"),
            dict(count=3, label="3h", step="hour", stepmode="backward"),
            dict(step="all")
        ])
    )
)

#Show
fig.show()

#Amazon

#Interval required 1 minute
#Amazon
Amazon = yf.download(tickers='AMZN', period='1d', interval='1m')


#declare figure
fig = go.Figure()

#Candlestick
fig.add_trace(go.Candlestick(x=Amazon.index,
                open=Amazon['Open'],
                high=Amazon['High'],
                low=Amazon['Low'],
                close=Amazon['Close'], name = 'market Amazon'))

# Add titles
fig.update_layout(
    title='Amazon live share price evolution',
    yaxis_title='Stock Price (USD per Shares)')

# X-Axes
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=15, label="15m", step="minute", stepmode="backward"),
            dict(count=45, label="45m", step="minute", stepmode="backward"),
            dict(count=1, label="HTD", step="hour", stepmode="todate"),
            dict(count=3, label="3h", step="hour", stepmode="backward"),
            dict(step="all")
        ])
    )
)

#Show
fig.show()


#Interval required 1 minute
#Disney
Disney = yf.download(tickers='DIS', period='1d', interval='1m')


#declare figure
fig = go.Figure()

#Candlestick
fig.add_trace(go.Candlestick(x=Disney.index,
                open=Disney['Open'],
                high=Disney['High'],
                low=Disney['Low'],
                close=Disney['Close'], name = 'market Disney'))

# Add titles
fig.update_layout(
    title='Disney live share price evolution',
    yaxis_title='Stock Price (USD per Shares)')

# X-Axes
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=15, label="15m", step="minute", stepmode="backward"),
            dict(count=45, label="45m", step="minute", stepmode="backward"),
            dict(count=1, label="HTD", step="hour", stepmode="todate"),
            dict(count=3, label="3h", step="hour", stepmode="backward"),
            dict(step="all")
        ])
    )
)

#Show
fig.show()


#Interval required 1 minute
#Google
Google = yf.download(tickers='GOOGL', period='1d', interval='1m')


#declare figure
fig = go.Figure()

#Candlestick
fig.add_trace(go.Candlestick(x=Google.index,
                open=Google['Open'],
                high=Google['High'],
                low=Google['Low'],
                close=Google['Close'], name = 'market Google'))

# Add titles
fig.update_layout(
    title='Google live share price evolution',
    yaxis_title='Stock Price (USD per Shares)')

# X-Axes
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=15, label="15m", step="minute", stepmode="backward"),
            dict(count=45, label="45m", step="minute", stepmode="backward"),
            dict(count=1, label="HTD", step="hour", stepmode="todate"),
            dict(count=3, label="3h", step="hour", stepmode="backward"),
            dict(step="all")
        ])
    )
)

#Show
fig.show()


#Interval required 1 minute
#Home Depot
HomeDept = yf.download(tickers='HD', period='1d', interval='1m')


#declare figure
fig = go.Figure()

#Candlestick
fig.add_trace(go.Candlestick(x=HomeDept.index,
                open=HomeDept['Open'],
                high=HomeDept['High'],
                low=HomeDept['Low'],
                close=HomeDept['Close'], name = 'market HomeDept'))

# Add titles
fig.update_layout(
    title='HomeDept live share price evolution',
    yaxis_title='Stock Price (USD per Shares)')

# X-Axes
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=15, label="15m", step="minute", stepmode="backward"),
            dict(count=45, label="45m", step="minute", stepmode="backward"),
            dict(count=1, label="HTD", step="hour", stepmode="todate"),
            dict(count=3, label="3h", step="hour", stepmode="backward"),
            dict(step="all")
        ])
    )
)

#Show
fig.show()


#Interval required 1 minute
#Walmart
Walmart = yf.download(tickers='WMT', period='1d', interval='1m')


#declare figure
fig = go.Figure()

#Candlestick
fig.add_trace(go.Candlestick(x=Walmart.index,
                open=Walmart['Open'],
                high=Walmart['High'],
                low=Walmart['Low'],
                close=Walmart['Close'], name = 'market Walmart'))

# Add titles
fig.update_layout(
    title='Walmart live share price evolution',
    yaxis_title='Stock Price (USD per Shares)')

# X-Axes
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=15, label="15m", step="minute", stepmode="backward"),
            dict(count=45, label="45m", step="minute", stepmode="backward"),
            dict(count=1, label="HTD", step="hour", stepmode="todate"),
            dict(count=3, label="3h", step="hour", stepmode="backward"),
            dict(step="all")
        ])
    )
)

#Show
fig.show()


#Interval required 1 minute
#Nike
Nike = yf.download(tickers='NKE', period='1d', interval='1m')


#declare figure
fig = go.Figure()

#Candlestick
fig.add_trace(go.Candlestick(x=Nike.index,
                open=Nike['Open'],
                high=Nike['High'],
                low=Nike['Low'],
                close=Nike['Close'], name = 'market Nike'))

# Add titles
fig.update_layout(
    title='Nike live share price evolution',
    yaxis_title='Stock Price (USD per Shares)')

# X-Axes
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=15, label="15m", step="minute", stepmode="backward"),
            dict(count=45, label="45m", step="minute", stepmode="backward"),
            dict(count=1, label="HTD", step="hour", stepmode="todate"),
            dict(count=3, label="3h", step="hour", stepmode="backward"),
            dict(step="all")
        ])
    )
)

#Show
fig.show()














