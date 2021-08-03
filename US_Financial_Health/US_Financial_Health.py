import codecademylib3_seaborn
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime
import pandas_datareader.wb as wb
import numpy as np

gold_prices = pd.read_csv('gold_prices.csv')
#print(gold_prices)

crude_oil_prices = pd.read_csv('crude_oil_prices.csv')
#print(crude_oil_prices)

start = datetime(1999, 1, 1) # year, month, day
end = datetime(2019, 1, 1)

nasdaq_data = web.DataReader('NASDAQ100', 'fred', start, end)
#print(nasdaq_data)

sap_data = web.DataReader('SP500', 'fred', start, end)
#print(sap_data)

gdp_data = wb.download(indicator='NY.GDP.MKTP.CD', country=['US'], start=start, end=end)
#print(gdp_data)

export_data = wb.download(indicator='NE.EXP.GNFS.CN', country=['US'], start=start, end=end)
#print(export_data)

def log_return(prices):
  return np.log(prices / prices.shift(1))

gold_returns = log_return(gold_prices['Gold_Price'])
crude_oil_returns = log_return(crude_oil_prices['Crude_Oil_Price'])
nasdaq_returns = log_return(nasdaq_data['NASDAQ100'])
sap_returns = log_return(sap_data['SP500'])
gdp_returns = log_return(gdp_data['NY.GDP.MKTP.CD'])
export_returns = log_return(export_data['NE.EXP.GNFS.CN'])

print("Comodity Volatility:")
print('Gold:', gold_returns.var())
print('Crude Oil:', crude_oil_returns.var())
print('NASDAQ:', nasdaq_returns.var())
print('S&P 500:', sap_returns.var())
print('GDP:', gdp_returns.var())
print('Export:', export_returns.var())

'''
SUMMARY OF FINDINGS:
The S&P 500, a collection of 500 large companies listed on stock exchanges in the United States, has the smallest variance, and thus is the least volatile. Given that the S&P 500 index is a weighted measurement of many stocks across a variety of industries, it is seen as a safer, diversified investment.

Gold, notorious for being a stable investment has the second smallest variance.

Crude oil is the most volatile, which makes sense as gas prices are often unpredictable, especially in the last 20 years.

The stocks are interesting. The NASDAQ 100 is more volatile than the S&P 500, which, when you think about it makes sense, as the S&P 500 is far more diversified and tracks more of the market.

Then finally we have GDP and exports.

Exports are very volatile, which could have to do with industries moving overseas in the last 20 years, and global competition for the production of goods.

GDP is actually fairly similar to the NASDAQ 100 in terms of volatility, which is perhaps an interesting correlation.
'''
