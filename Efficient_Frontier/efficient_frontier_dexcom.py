import pandas as pd
import matplotlib.pyplot as plt
from rf import *
#import codecademylib3_seaborn
import seaborn
import numpy as np

#path='stock_data.csv'
#path='stock_data_nvidia.csv'
path='stocks_nvidia_dexcom.csv'

stock_data = pd.read_csv(path)
selected=list(stock_data.columns[1:])

returns_quarterly = stock_data[selected].pct_change()
expected_returns = returns_quarterly.mean()
cov_quarterly = returns_quarterly.cov()

single_asset_std=np.sqrt(np.diagonal(cov_quarterly))
df = return_portfolios(expected_returns, cov_quarterly) 
weights, returns, risks = optimal_portfolio(returns_quarterly[1:])

df.plot.scatter(x='Volatility', y='Returns', fontsize=12)
plt.plot(risks, returns, 'y-o')
plt.scatter(single_asset_std,expected_returns,marker='X',color='red',s=200)


if 'dexcom' in path:
  plt.axvline(single_asset_std[-1], color='black')
  plt.scatter(single_asset_std[-1],expected_returns[-1],marker='X',color='black',s=200)
  plt.scatter(single_asset_std[-2],expected_returns[-2],marker='X',color='green',s=200)
  original_EF=pd.read_csv('./risks_returns_original.csv')
  nvidia_EF=pd.read_csv('./risks_returns_nvidia.csv')
  plt.plot(risks, returns, 'k-o')
  plt.plot(original_EF['risks'],original_EF['returns'], 'y-o')
  plt.plot(nvidia_EF['risks'],nvidia_EF['returns'], 'g-o')
elif 'nvidia' in path:
  plt.axvline(single_asset_std[-1], color='green')
  plt.scatter(single_asset_std[-1],expected_returns[-1],marker='X',color='green',s=200)
  original_EF=pd.read_csv('./risks_returns_original.csv')
  plt.plot(risks, returns, 'g-o')
  plt.plot(original_EF['risks'],original_EF['returns'], 'y-o')
  
plt.ylabel('Expected Returns',fontsize=14)
plt.xlabel('Volatility (Std. Deviation)',fontsize=14)
plt.title('Efficient Frontier', fontsize=24)
plt.show()

'''
In this example, a high-risk, high-return asset increased the expected return of portfolios on the efficient frontier. This is because the new asset had a larger expected return than any other asset.

Now let’s think about how we can decrease the risk of portfolios on the efficient frontier. What kind of asset do you think we should choose to decrease the risk of portfolios on the efficient frontier?

To decrease the risk of portfolios on an efficient frontier, you should add assets that are uncorrelated to assets in your portfolio.

Unlike with expected return, the risk of an individual asset does not indicate whether it will affect the risk of the overall portfolio. An uncorrelated asset increases the diversity of your portfolio, which protects you from a downturn that may affect a subset of your assets.
'''