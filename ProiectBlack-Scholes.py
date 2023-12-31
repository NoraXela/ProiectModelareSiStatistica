from math import log, sqrt, pi, exp
from scipy.stats import norm
from datetime import datetime, date
import numpy as np
import pandas as pd
from pandas import DataFrame
import pandas_datareader.data as web
import random
import matplotlib.pyplot as plt

# d1 si d2
def d1(S,K,T,r,sigma):
    return(log(S/K)+(r+sigma**2/2.)*T)/(sigma*sqrt(T))
def d2(S,K,T,r,sigma):
    return d1(S,K,T,r,sigma)-sigma*sqrt(T)

# functii call si put
def bs_call(S,K,T,r,sigma):
    return S*norm.cdf(d1(S,K,T,r,sigma))-K*exp(-r*T)*norm.cdf(d2(S,K,T,r,sigma))
  
def bs_put(S,K,T,r,sigma):
    return K*exp(-r*T)-S*bs_call(S,K,T,r,sigma)

# implied volatility
def call_implied_volatility(Price, S, K, T, r):
    sigma = 0.001
    while sigma < 1:
        Price_implied = S * \
            norm.cdf(d1(S, K, T, r, sigma))-K*exp(-r*T) * \
            norm.cdf(d2(S, K, T, r, sigma))
        if Price-(Price_implied) < 0.001:
            return sigma
        sigma += 0.001
    return "Not Found"

def put_implied_volatility(Price, S, K, T, r):
    sigma = 0.001
    while sigma < 1:
        Price_implied = K*exp(-r*T)-S+bs_call(S, K, T, r, sigma)
        if Price-(Price_implied) < 0.001:
            return sigma
        sigma += 0.001
    return "Not Found"

# collecting data
stock = 'SPY'
expiry = '02-18-2024'
strike_price = 370

today = datetime.now()
# one_year_ago = today.replace(year=today.year-1)


# print(one_year_ago)
# print(today)
print('testx')

# df = web.DataReader(name=stock, data_source='yahoo', start=one_year_ago, end=today)

#INCERCARE!!!
start = pd.to_datetime(['2007-01-01']).astype(int)[0]//10**9 # convert to unix timestamp.
end = pd.to_datetime(['2020-12-31']).astype(int)[0]//10**9 # convert to unix timestamp.
url = 'https://query1.finance.yahoo.com/v7/finance/download/' + stock + '?period1=' + str(start) + '&period2=' + str(end) + '&interval=1d&events=history'
df = pd.read_csv(url)
print(df)
#END-INCERCARE

print('test2')

df = df.sort_values(by="Date")
df = df.dropna()
df = df.assign(close_day_before=df.Close.shift(1))
df['returns'] = ((df.Close - df.close_day_before)/df.close_day_before)

print('test')

sigma = np.sqrt(252) * df['returns'].std()
# sigma = 0.90
print('sigma ', sigma)
#   INCERCARE PE uty
# uty = (web.DataReader(
#     "^TNX", 'yahoo', today.replace(day=today.day-1), today)['Close'].iloc[-1])/100
# url2 = 'https://query1.finance.yahoo.com/v7/finance/download/^TNX?period1=' + str(start) + '&period2=' + str(end) + '&interval=1d&events=history'
# uty = (pd.read_csv(url2))/100
uty = 4.046/100
print('uty ', uty)
#   FINAL INCERCARE PE uty

lcp = df['Close'].iloc[-1]
print('test3')
t = (datetime.strptime(expiry, "%m-%d-%Y") - datetime.utcnow()).days / 365
print('test4')
print('* * *')
print('The Option Price is: ', bs_call(lcp, strike_price, t, uty, sigma))
# print('lcp ', lcp)
# print('strike_price ', strike_price)
# print('t ', t)
# print('uty ', uty)
# print('sigma ', sigma)
# print(bs_call(lcp, strike_price, t, uty, sigma))
#print('test5')
print("Implied Volatility: " +
      str(100 * call_implied_volatility(bs_call(lcp, strike_price, t, uty, sigma,), lcp, strike_price, t, uty,)) + " %")
#print('test6')

#Parte noua
xArr = []
yArr = []
for i in range(1000):
    h = random.randrange(0, int(sigma*10000))
    sigmaR = h/10000
    xArr.append(sigmaR)
#     sigmaR = np.random.uniform(0.0, sigma)
#     # while sigmaR < 0:
#     #     sigmaR = np.random.uniform(0.0, sigma)
    print('* * *')
# #print("sigma", sigma)   
    print("sigmaR", sigmaR)   
    print('The Option Price is: ', bs_call(lcp, strike_price, t, uty, sigmaR))
    yArr.append(bs_call(lcp, strike_price, t, uty, sigmaR))
    print("Implied Volatility sigmaR: " +
      str(100 * call_implied_volatility(bs_call(lcp, strike_price, t, uty, sigmaR,), lcp, strike_price, t, uty,)) + " %")   

print("xArr ", xArr)
print("yArr ", yArr)

# plot
sizes = np.random.uniform(15, 80, len(xArr))
colors = np.random.uniform(15, 80, len(xArr))
fig, ax = plt.subplots()
ax.scatter(xArr, yArr, s=sizes, c=colors, vmin=0, vmax=100)
# ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
#        ylim=(0, 1), yticks=np.arange(1, 8))
plt.show()

# for i in range(10):  # Plotting 10 random paths for illustration
#     #plt.plot(np.arange(0, T + dt, dt), stock_prices[i, :])

# h = random.randrange(0, 2000)
# j = h/10000
# print("h",h)
# print("j",j)
                    
