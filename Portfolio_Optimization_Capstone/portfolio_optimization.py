"""
Created on Tue Aug 22 15:00:17 2021
@author: Scott Burstein

This project was completed as the Capstone project for Codecademy's Analyze Financial Data With Python Skill Path.
The portfolio optimization aims to identify an optimal portfolio of 4 selected stocks: JPM, GOOG, UL, and JNJ.
The project also demonstrates an efficient frontier to show risk-return relationships 
for 10,000 portfolios with randomly allocated weights of these 4 stocks.

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
import cvxopt as opt
from cvxopt import blas, solvers

def return_portfolios(expected_returns, cov_matrix):
    port_returns = []
    port_volatility = []
    stock_weights = []
    
    selected = (expected_returns.axes)[0]
    
    num_assets = len(selected) 
    num_portfolios = 10000
    
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
  
  
def optimal_portfolio(returns):
    n = returns.shape[1]
    returns = np.transpose(returns.to_numpy())

    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x']
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks

tickers = yf.Tickers(['GOOG', 'JPM', 'UL', 'JNJ'])
start = datetime(2016,1,1)
end = datetime(2021,1,1)
#weekly_prices = tickers.history(period = '5Y', interval = '1wk')
weekly_prices = pd.read_csv('weekly_prices.csv')
print(weekly_prices.head())
selected = list(weekly_prices.columns[1:])
weekly_returns = weekly_prices[selected].pct_change()
print(weekly_returns.head())

mean_returns = weekly_returns.mean()
cov_matrix = weekly_returns.cov()

port_return = return_portfolios(mean_returns, cov_matrix)
print(port_return.head())
weights, returns, risks = optimal_portfolio(weekly_returns[1:])

plt.scatter(port_return.Volatility, port_return.Returns)
plt.plot(risks,returns,'y-o')
ax = plt.subplot()
ax.set_xlabel('Risk')
ax.set_ylabel('Expected Return')
plt.title('Risks vs Returns of Random portfolios')