import pandas as pd
import matplotlib.pyplot as plt
from rf import return_portfolios, optimal_portfolio
#import codecademylib3_seaborn
import seaborn
import numpy as np

path='stock_data3.csv'

# 1. Load the stock data
stock_data = pd.read_csv(path)
#stock_names = ['PFE', 'TGT', 'M', 'VZ', 'JPM', 'MRO', 'KO', 'PG', 'CVS', 'HPQ']
# SOLUTION:
selected = ['JPM', 'VZ', 'CVS', 'TGT', 'PFE']
#selected = ['PFE', 'TGT', 'M', 'VZ', 'JPM', 'MRO', 'KO', 'PG', 'CVS', 'HPQ']

# 2. Find the quarterly for each period
returns_quarterly = stock_data[selected].pct_change()

# 3. Find the expected returns 
expected_returns = returns_quarterly.mean()
print(expected_returns)

# 4. Find the covariance 
cov_quarterly = returns_quarterly.cov()
print(cov_quarterly)

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

#REMOVE THIS:
#pd.DataFrame({'Risks': risks, 'Returns': returns}).to_csv('all_ten.csv', index=False)

# 9. Compare the set of portfolios on the EF to the 
single_asset_std=np.sqrt(np.diagonal(cov_quarterly))
plt.scatter(single_asset_std,expected_returns,marker='X',color='red',s=200)

# All 10
all_ten_EF = pd.read_csv('all_ten.csv')
plt.plot(all_ten_EF['Risks'], all_ten_EF['Returns'], 'g-o')
plt.show()


'''
Using the following five assets will result in an efficient frontier that closely resembles a portfolio with all ten assets.

selected = ['TGT', 'CVS', 'M', 'VZ', 'JPM']
We used the following approach to select these assets:

JPM has the largest expected return, so we knew to include it.
We added the asset that was the least correlated to JPM – VZ has a small negative covariance.
We selected the assets with the next three highest expected returns: M, CVS, and TGT.
'''