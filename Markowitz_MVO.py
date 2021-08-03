import pandas as pd 
from matplotlib import pyplot as plt
import codecademylib3_seaborn
import numpy as np
import cvxopt as opt
from cvxopt import blas, solvers

path = 'stock_data.csv'

stock_data = pd.read_csv(path)
selected = list(stock_data.columns[1:])

returns_quarterly = stock_data[selected].pct_change()
expected_returns = returns_quarterly.mean()
cov_quarterly = returns_quarterly.cov()

def optimal_portfolio(returns):
  n = returns.shape[1]
  returns = np.transpose(returns.as_matrix())

  N = 100
  mus = [10**(5.0 * t/N - 1.0) for t in range(N)]

  S = opt.matrix(np.cov(returns))
  pbar = opt.matrix(np.mean(returns, axis=1))

  G = -opt.matrix(np.eye(n))   
  h = opt.matrix(0.0, (n ,1))
  A = opt.matrix(1.0, (1, n))
  b = opt.matrix(1.0)

  portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x']
                for mu in mus]

  returns = [blas.dot(pbar, x) for x in portfolios]
  risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]

  m1 = np.polyfit(returns, risks, 2)
  x1 = np.sqrt(m1[2] / m1[0])

  wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
  return np.asarray(wt), returns, risks


def return_portfolios(expected_returns, cov_matrix):
  port_returns = []
  port_volatility = []
  stock_weights = []
  
  num_assets = len(selected) 
  num_portfolios = 5000
  
  for single_portfolio in range(num_portfolios):
      weights = np.random.random(num_assets)
      weights /= np.sum(weights)
      returns = np.dot(weights, expected_returns)
      volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
      port_returns.append(returns)
      port_volatility.append(volatility)
      stock_weights.append(weights)
  
  portfolio = {'Returns': port_returns,
               'Volatility': port_volatility}
  
  for counter,symbol in enumerate(selected):
      portfolio[symbol +' Weight'] = [Weight[counter] for Weight in stock_weights]
  
  df = pd.DataFrame(portfolio)
  
  column_order = ['Returns', 'Volatility'] + [stock+' Weight' for stock in selected]
  
  df = df[column_order]
 
  return df

df = return_portfolios(expected_returns, cov_quarterly) 

weights, returns, risks = optimal_portfolio(returns_quarterly[1:])

min_risk_idx = np.array(risks).argmin()
max_return_idx = np.array(risks).argmax()
min_risk = [risks[min_risk_idx], returns[min_risk_idx]]
max_return = [risks[max_return_idx], returns[max_return_idx]]

df.plot.scatter(x='Volatility', y='Returns', fontsize=12)
plt.plot(risks, returns, 'y-o')
plt.plot(min_risk[0],min_risk[1],'r^', markersize=12)
plt.plot(max_return[0],max_return[1],'rX',markersize=16)
plt.ylabel('Expected Returns',fontsize=20)
plt.xlabel('Volatility (Std. Deviation)',fontsize=20)
plt.title('Efficient Frontier', fontsize=24)
plt.show()
