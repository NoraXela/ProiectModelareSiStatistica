from math import log, sqrt, exp
from scipy.stats import norm
from datetime import datetime
import numpy as np
import pandas as pd
import random

# Black-Scholes formulas
def d1(S, K, T, r, sigma):
    return (log(S / K) + (r + sigma**2 / 2.) * T) / (sigma * sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * sqrt(T)

def bs_call(S, K, T, r, sigma):
    return S * norm.cdf(d1(S, K, T, r, sigma)) - K * exp(-r * T) * norm.cdf(d2(S, K, T, r, sigma))

# Collecting data
stock = 'SPY'
expiry = '02-18-2024'
strike_price = 370

# Download historical stock data
start_date = pd.to_datetime('2007-01-01')
end_date = pd.to_datetime('2020-12-31')
url = f'https://query1.finance.yahoo.com/v7/finance/download/{stock}?period1={int(start_date.timestamp())}&period2={int(end_date.timestamp())}&interval=1d&events=history'
df = pd.read_csv(url)
print (df)

# Preprocess data
df = df.sort_values(by="Date")
df = df.dropna()
df = df.assign(close_day_before=df['Close'].shift(1))
df['returns'] = (df['Close'] - df['close_day_before']) / df['close_day_before']

# Calculate volatility
sigma = np.sqrt(252) * df['returns'].std()

# Calculate other Black-Scholes parameters
today = datetime.now()
lcp = df['Close'].iloc[-1]
t = (datetime.strptime(expiry, "%m-%d-%Y") - today).days / 365
uty = 4.046 / 100

# Calculate and print the Black-Scholes call option price
option_price = bs_call(lcp, strike_price, t, uty, sigma)
print(f'Pretul optiunii : {option_price}')

#Generare grafic de tip scatter cu sigma random
random_volatilities = np.random.uniform(0, 0.2, 100)
option_prices_random_vol = [bs_call(lcp, strike_price, t, uty, sigma_r) for sigma_r in random_volatilities]

import matplotlib.pyplot as plt

plt.scatter(random_volatilities, option_prices_random_vol, color='blue')
plt.title('Pretul Call-ului in functie de Sigma')
plt.xlabel('Sigma')
plt.ylabel('Pretul Call-ului')
plt.show()