# 定期再平衡
> 將台股資料匯入，並執行策略，來看最終的績效如何。  
此專案依每此再平衡之間隔分成 [年平衡](https://github.com/JT-427/auto-rebalance-TW/blob/master/requirements.txt)、[季平衡](https://github.com/JT-427/auto-rebalance-TW/blob/master/requirements.txt)

## Requirements
- python >= 3.8
- [requirements.txt](https://github.com/JT-427/auto-rebalance-TW/blob/master/requirements.txt)

## Install package
```sh
pip install -r requirements.txt
```

## Run
```sh
cd <folder> # 'annual rebalance', 'quarterly rebalance'
python run.py
```

## Description
### Duration
&nbsp; 2011/1~今天

### Strategy
1. 選股池  
&nbsp; 台股市值前150

2. 利用回歸模型，計算出各檔股票的統計數據（資料來源為股價日報酬）  
    ```py
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
    ```


3. 挑選標的
&nbsp; 挑選Alpha最大的前五檔股票來投資
    ```py
    choose = Stock_stat.sort_values('Alpha',ascending = False).head(5)
    ```

4. 配權重  
&nbsp; 用Markowitz Efficient Frontier來取得權重

    ```py
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

    '''weights'''
    mean_return = C_return.mean()
    cov_matrix = C_return.cov()
    wT = min_variance(mean_return,cov_matrix)['x']
    vol = Portfolio_volatility(wT,mean_return,cov_matrix)
    print('weights:',wT)
    print('volatility:',vol)
    ''''''
    ```

5. 定期再平衡  
&nbsp; 每 年or季 會執行一次再平衡，將所有的股票釋出，重新挑選標的買進。


### Result
投資組合與大盤和0050之價值比較
||績效比較|
|--|--|
|年|![img](https://github.com/JT-427/regularly-rebalance-TW/blob/master/annual%20rebalance/Output/ANNUAL_Portfolio%20vs%20TWII%20vs%200050.png)|
|季|![img](https://github.com/JT-427/regularly-rebalance-TW/blob/master/quarterly%20rebalance/Output/Quarter_Portfolio%20vs%20TWII%20vs%200050.png)|


## Resource
- [yfinance](https://github.com/ranaroussi/yfinance)


***
### 專案開發期間
2019/11 ~ 2019/12
