import pandas as pd
import matplotlib.pyplot as plt
from rf import return_portfolios, optimal_portfolio
#import codecademylib3_seaborn
import seaborn
import numpy as np

path='stock_data_weak.csv'

# 1. Load the stock data
stock_data = pd.read_csv(path)
print(stock_data)

# 2. Find the quarterly return for each period
selected = list(stock_data.columns[1:])
returns_quarterly = stock_data[selected].pct_change()

# 3. Find the expected returns 
expected_returns = returns_quarterly.mean()

# 4. Find the covariance 
cov_quarterly = returns_quarterly.cov()

# 5. Find a set of random portfolios
random_portfolios = return_portfolios(expected_returns, cov_quarterly) 

# 6. Plot the set of random portfolios
random_portfolios.plot.scatter(x='Volatility', y='Returns', fontsize=12)

# 7. Calculate the set of portfolios on the EF
weights, returns, risks = optimal_portfolio(returns_quarterly[1:])

# 8. Plot the set of portfolios on the EF
plt.plot(risks, returns, 'y-o')
plt.ylabel('Expected Returns',fontsize=14)
plt.xlabel('Volatility (Std. Deviation)',fontsize=14)
plt.title('Efficient Frontier', fontsize=24)

# Compare the set of portfolios on the EF to the 
try:
  single_asset_std=np.sqrt(np.diagonal(cov_quarterly))
  plt.scatter(single_asset_std,expected_returns,marker='X',color='red',s=200)
except:
  pass
plt.show()