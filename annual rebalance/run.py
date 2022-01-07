#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 16:10:58 2020

@author: jimmytseng
"""

import datetime as dt
starttime = dt.datetime.now() #開始時間
##############################################################################
import sys
import os
path = os.getcwd()
src = os.path.join(path,'Input')
out = os.path.join(path,'Output')

from module.tw_stock import get_data
import pandas as pd
import numpy as np

#=============================================================================
'''股票資料'''
os.chdir(os.path.join(path,'module/tw_stock'))
df = get_data.get()
''''''
#=============================================================================
''''''

'''
==============================================================================
'''

''''''
#=============================================================================
'''開始投資日期＆前半年日期'''
def podt(place_order_date):
    #p_date = dt.date(y, m, d)
    p_date = place_order_date
    if p_date.month <= 5 :
        year = p_date.year - 1
        month = p_date.month + 6
        d_date = dt.date(year, month, p_date.day)
        return d_date
    if p_date.month > 5 :
        year = p_date.year
        month = p_date.month - 6
        d_date = dt.date(year, month, p_date.day)
        return d_date
''''''
#=============================================================================
'''選定期間之股票調整後收盤價'''
def StockD(d1,d2):
    Stock_data=df[d1 : d2]
    
    Stock_data = Stock_data['Adj Close']
    Stock_data = Stock_data.dropna(how = 'all')
    Stock_data = Stock_data.fillna(method='ffill')
    Stock_data = Stock_data.dropna(axis= 1)
    return Stock_data
''''''
#=============================================================================
'''return'''
def ret(Stock_data,N):
    return_Stock_data = np.log(Stock_data/Stock_data.shift(N))
    return_Stock_data = return_Stock_data.fillna(method='ffill')
    return_Stock_data = return_Stock_data.dropna(how = 'all')
    return return_Stock_data
''''''
#=============================================================================
'''Markowitz'''
# return and volatility functions
def Portfolio_performance(weights, mean_returns, cov_matrix):
	returns = np.sum(mean_returns *weights )
	std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
	return std, returns
# volatility function of portfolio
def Portfolio_volatility(weights, mean_returns, cov_matrix):
	return Portfolio_performance(weights, mean_returns, cov_matrix)[0]
# MV model
def min_variance(mean_returns, cov_matrix):
	num_assets = len(mean_returns)
	args = (mean_returns, cov_matrix)
	constraints = ({'type':'eq', 'fun': lambda x: np.sum(x)-1})
	bound = (0,1)
	bounds = tuple(bound for asset in range(num_assets))
	result = minimize(Portfolio_volatility, num_assets *[1/num_assets , ], args=args ,
                   method= 'SLSQP' , bounds=bounds , constraints=constraints )
	return result
''''''
#=============================================================================
''''''

'''
==============================================================================
'''

''''''
#=============================================================================
''''''
os.chdir(out)
import statsmodels.api as sm
import scipy.stats as si
from scipy.optimize import minimize
import matplotlib.pyplot as plt

year = list(np.unique(df.index.year))
# start_y = int(input('輸入開始投資年：'))
start_y = 2011

year = year[year.index(start_y):]


wTs = pd.DataFrame()

Value = {'year':['initial']+year, 'Portfolio':[1000], 'TWII':[1000],'0050':[1000]}


for i in range(len(year)):
    y = i + year[0]

    p_date = dt.date(y,1,1)
    # d_date = dt.date(y-1,1,1)
    d_date = podt(p_date)
    print('迴歸期間：',d_date,'~',p_date+dt.timedelta(days=-1))
    
    Stock_data = StockD(d_date ,p_date+dt.timedelta(days=-1))
    
    StocksList = list(Stock_data.columns)
    ''''''
    #=============================================================================
    ''''''
    N = 1
    return_Stock_data = ret(Stock_data,N)
    ''''''
    #=============================================================================
    '''迴歸'''
    XX=sm.add_constant(return_Stock_data['^TWII'])
    
    lr_list=[]
    A = {}
    l_alpha = []
    l_alpha_P = []
    l_beta = []
    l_beta_P = []
    for g in StocksList:
        est = sm.OLS(return_Stock_data[g] , XX)
        est = est.fit()
        sol = est.summary()
        lr_list.append(sol.as_csv())
        
        An = pd.DataFrame(est.params)
        An['coeff'] = est.params
        An["p_values"] = est.summary2().tables[1]["P>|t|"]
        An = An.iloc[:, [1, 2]]
        An.rename(index = {'const' : 'Alpha','^TWII' : 'Beta'},inplace = True)
        A[g] = An
        
        l_alpha.append(A[g].iloc[0,0]) #alpha
        l_alpha_P.append(A[g].iloc[0,1]) #alpha P-value
        l_beta.append(A[g].iloc[1,0]) #beta
        l_beta_P.append(A[g].iloc[1,1]) #beta
    ''''''
    #=============================================================================
    ''''''
    mean = return_Stock_data.mean()
    std = return_Stock_data.std()
    # corr = return_Stock_data.corr()
    skew = pd.Series(si.skew(Stock_data), index = (StocksList))
    kurt = pd.Series(si.kurtosis(Stock_data), index = (StocksList))
    alpha = pd.Series(l_alpha, index = (StocksList))
    alpha_P = pd.Series(l_alpha_P, index = (StocksList))
    beta = pd.Series(l_beta, index = (StocksList))
    beta_P = pd.Series(l_beta_P, index = (StocksList))
    
    Stock_stat = pd.concat([mean, std, skew, kurt, alpha, alpha_P, beta, beta_P],axis = 1)
    Stock_stat.columns = ['mean', 'std', 'skew', 'kurt', 'Alpha', 'alpha_P', 'Beta', 'beta_P']
    Stock_stat.to_excel('Stock_stat.xlsx')
    ''''''
    #=============================================================================
    
    
    '''
    ==============================================================================
    '''
    
    
    #=============================================================================
    '''選五檔股票'''
    choose = Stock_stat.sort_values('Alpha',ascending = False).head(5)
    # choose_5 = Stock_stat.sort_values('skew',ascending = False).head(10)
    # choose_5 = Stock_stat.sort_values('Beta',ascending = True).head(5)
    
    choose = list(choose.index)
    print('choose:',choose)
    
    C_return = return_Stock_data[choose]
    ''''''
    #=============================================================================
    '''weights'''
    mean_return = C_return.mean()
    cov_matrix = C_return.cov()
    wT = min_variance(mean_return,cov_matrix)['x']
    vol = Portfolio_volatility(wT,mean_return,cov_matrix)
    print('weights:',wT)
    print('volatility:',vol)
    ''''''
    #=============================================================================
    ''''''
    print('投資期間：',p_date.year)
    PStock_data = df[str(p_date.year)]
    PStock_data = PStock_data['Adj Close']
    PStock_data = PStock_data.dropna(how = 'all')
    PStock_data = PStock_data.fillna(method='ffill')
    PStock_data = PStock_data.dropna(axis= 1)
    
    '''choose'''
    CStock_data = PStock_data[choose]
    
    buyprice = CStock_data.iloc[0]
    sellprice = CStock_data.iloc[-1]
    
    unit = Value['Portfolio'][i]*wT/buyprice
    spread = sellprice - buyprice
    Value['Portfolio'].append(Value['Portfolio'][i]+(spread*unit).sum())
    ''''''
    '''TWII'''
    PTWII = PStock_data['^TWII']
    
    T_buyprice = PTWII.iloc[0]
    T_sellprice = PTWII.iloc[-1]
    
    T_unit = Value['TWII'][i]/T_buyprice
    T_spread = T_sellprice - T_buyprice
    Value['TWII'].append(Value['TWII'][i]+T_spread*T_unit)
    ''''''
    '''0050'''
    P_0050 = PStock_data['0050.TW']
    
    O_buyprice = P_0050.iloc[0]
    O_sellprice = P_0050.iloc[-1]
    
    O_unit = Value['0050'][i]/O_buyprice
    O_spread = O_sellprice - O_buyprice
    Value['0050'].append(Value['0050'][i]+O_spread*O_unit)
    ''''''
    ''''''
    print('===========================================================')
    
    ''''''

Value = pd.DataFrame(Value)
for i in range(len(Value)):
    Value['year'][i] = str(Value['year'][i])

plt.figure(figsize=(16, 9),dpi=500)
plt.plot( 'year', 'Portfolio' , data = Value, color = '#f01111', linewidth=2)
plt.plot( 'year', 'TWII', data = Value, color = '#5DADE2', linewidth=2)
plt.plot( 'year', '0050', data = Value, color = '#5de4b5', linewidth=2)
plt.legend(prop={'size':10})
plt.grid(True)
plt.title( 'ANNUAL\nPortfolio vs TWII vs 0050',fontsize=18)
plt.xlabel( 'Year' ,fontsize=15)
plt.ylabel( 'Value' ,fontsize=15)
plt.savefig('ANNUAL_Portfolio vs TWII vs 0050.png' )

Value.set_index('year',inplace=True)
Value.to_csv('ANNUAL_Portfolio vs TWII vs 0050.csv',index = True)

print(Value)

IRR = np.log(Value.iloc[len(Value)-1]/Value.iloc[0])/len(Value)

return_Value = ret(Value,1)
ER = return_Value.mean()
sigma = return_Value.std()
rf = 0.05
Sharp_Ratio = (ER - rf)/sigma

Performance = pd.concat([IRR,sigma,Sharp_Ratio],axis=1)
Performance.rename(columns={0:'IRR',1:'sigma',2:'Shape_Ratio'},inplace=True)
Performance.to_csv('Performance.csv')


##############################################################################
endtime = dt.datetime.now() #結束時間
print(endtime - starttime) #程式執行時間



