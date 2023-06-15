import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def fetch_data(symbol, interval, start, end):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={start}&endTime={end}'
    data = requests.get(url).json()
    df = pd.DataFrame(data)
    df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'num_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['open'] = pd.to_numeric(df['open'])
    df['close'] = pd.to_numeric(df['close'])
    return df[['open_time', 'open', 'close']]

def calculate_returns(df):
    df['return'] = (df['close'] / df['open']) - 1
    df = df[(df['return'].abs() <= 0.25)] # filter returns greater than 25%
    df['return'] = df['return'] * 100 # convert to percentage
    return df

start_time = pd.Timestamp('2018-06-15').value // 10**6  # convert to milliseconds
end_time = pd.Timestamp('2023-06-15').value // 10**6  # convert to milliseconds
data = fetch_data('ETHUSDT', '1d', start_time, end_time)

returns = calculate_returns(data)

def plot_distribution(df):
    data = df['return'].values
    data = data[~np.isnan(data)] # remove NaN values

    mu, std = norm.fit(data)

    plt.figure(figsize=(10,6))
    plt.hist(data, bins=100, density=True, alpha=0.6, color='g', edgecolor='black')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r', linewidth=2)
    plt.title('Normal Distribution of ETH/USDT Daily Returns', fontsize=15)
    plt.xlabel('Returns (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    plt.savefig('financialdatas.png')

plot_distribution(returns)
