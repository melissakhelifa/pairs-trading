# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 17:31:46 2021

@author: Mélissa
"""
#import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  itertools import combinations

# Import data from csv files
# Dataset    
all_data = pd.read_csv (r'.\data\data_mef.csv', delimiter=';')
# Inputs of the model
inputs_data = pd.read_csv (r'.\data\inputs_data.csv', delimiter=';')

### Cleaning of the dataset ### 
def clean_dataset(dataset):
    print("\nCleaning the dataset...\n")
    clean_data = dataset.copy()
    #Clean work #1: Standardize dataset so that at t=0, all values from all variables are valid (get rid of nan)
    noms = clean_data.columns
    values = []
    
    #For each variable, we find the index where the first valid value is available and stock them in a list
    for i in range(len(noms)-1):
        values.append(clean_data[noms[i]].first_valid_index())

    #Define the max index value so that values are valid for all variables
    all_valid_index = max(values)

    #Redefine the dataset
    if(all_valid_index>0):
        clean_data.drop(clean_data.loc[0:all_valid_index].index, inplace=True)
        clean_data = clean_data.reset_index(drop=True)

    #Clean work #2: Fill NA with the last valid observation
    clean_data = clean_data.fillna(method="ffill")
    
    #Clean work #3: Convert string to dates with Datetime 
    clean_data[noms[0]] = pd.to_datetime(clean_data[noms[0]],dayfirst=True) #Dates should be first column
    clean_data[noms[0]] = pd.to_datetime(clean_data[noms[0]]).apply(lambda x: x.date())

    print("Dataset cleaned.\n")
    return clean_data

### Split dataset into train et test datasets ### 
def split_train_test(dataset,percent_train):
    print("Splitting the dataset...\n")
    last_index_train = round(percent_train*len(dataset))

    data_train = dataset.iloc[0:last_index_train]
    data_test = dataset.iloc[last_index_train+1:len(dataset)]
    
    #Set Dates as Index
    data_train = data_train.set_index(dataset.columns[0])
    data_test = data_test.set_index(dataset.columns[0])
    
    print("Dataset split.\n")
    return data_train, data_test

### Calculate Momentum signals ###
def momentum_signals(dataset, mom_type, mom_duree):
    print("Calculating the momentum signals...\n") 
    
    nb_bdays_per_month = 21
    nb_bdays_per_year = nb_bdays_per_month*12
    convert_duree_month = mom_duree * nb_bdays_per_month
    convert_duree_year = mom_duree * nb_bdays_per_year
    limit_monthly_range = nb_bdays_per_month + convert_duree_month
    limit_yearly_range = nb_bdays_per_month + convert_duree_year
        
    #For each calculation, we consider a 1m lag
    if mom_type == "d" or mom_type == "j":
        row = len(dataset) - mom_duree + nb_bdays_per_month
        col = len(dataset.columns)
        momentum_signal = [[0 for x in range(row - nb_bdays_per_month)] for y in range(col)]
    
        for i in range(col):
            for j in range(nb_bdays_per_month,row):                  
                momentum_signal[i][j-nb_bdays_per_month]= np.log(dataset.iloc[j+mom_duree,i]/dataset.iloc[j,i])
            
    elif mom_type == "m":
        row = len(dataset)-limit_monthly_range + nb_bdays_per_month
        col = len(dataset.columns)
        momentum_signal = [[0 for x in range(row - nb_bdays_per_month)] for y in range(col)]  
        
        for i in range(col):
            for j in range(nb_bdays_per_month,row):   
                momentum_signal[i][j-nb_bdays_per_month]=np.log(dataset.iloc[j+convert_duree_month,i]/dataset.iloc[j,i])
        
    elif mom_type == "y"or mom_type == "a":
        row = len(dataset)-limit_yearly_range + nb_bdays_per_month
        col = len(dataset.columns)
        momentum_signal = [[0 for x in range(row - nb_bdays_per_month)] for y in range(col)]
        
        for i in range(col):    
            for j in range(nb_bdays_per_month,row):
                momentum_signal[i][j-nb_bdays_per_month]=np.log(dataset.iloc[j+convert_duree_year,i]/dataset.iloc[j,i])
    
    #Convert list to dataframe with same properties as the original
    momentum_signal = pd.DataFrame(momentum_signal)    
    momentum_signal = momentum_signal.transpose()      
    momentum_signal.columns=dataset.columns.values
    if mom_type == "d" or mom_type == "j":
        momentum_signal=momentum_signal.set_index(dataset.index[nb_bdays_per_month+mom_duree:len(dataset)])
    elif mom_type == "m":
        momentum_signal=momentum_signal.set_index(dataset.index[nb_bdays_per_month+convert_duree_month:len(dataset)])
    elif mom_type == "y"or mom_type == "a":
        momentum_signal=momentum_signal.set_index(dataset.index[nb_bdays_per_month+convert_duree_year:len(dataset)])
    
    print("Momentum signals calculated.\n")  
    return momentum_signal

### Calculate returns ###
def returns(indices,frequency):
       
    if(frequency=="d")or(frequency=="j"):
        print("Calculating indices daily returns...\n")
        daily_returns = indices.copy()
        daily_returns = daily_returns.pct_change()
        daily_returns = daily_returns.iloc[1: , :]
        
        print("Indices daily returns calculated.\n")
        return daily_returns
    
    elif(frequency=="w")or(frequency=="s"):
        print("Calculating indices weekly returns...\n")
        weekly_returns = pd.DataFrame(columns = indices.columns)
        append_data = []
        
        # for loop with an incrementation of 5 => we take the data every week
        for i in range(0, len(indices), 5):
            append_data.append(indices.iloc[[i]])
        weekly_returns = pd.concat(append_data)
        weekly_returns = weekly_returns.pct_change()
        weekly_returns = weekly_returns.iloc[1: , :]
        
        print("Indices monthly weekly calculated.\n")
        return weekly_returns
    
    elif(frequency=="m"):
        print("Calculating indices monthly returns...\n")
        monthly_returns = pd.DataFrame(columns = indices.columns)
        append_data = []
        
        # for loop with an incrementation of 21 => we take the data every month
        for i in range(0, len(indices), 21):
            append_data.append(indices.iloc[[i]])
        monthly_returns = pd.concat(append_data)
        monthly_returns = monthly_returns.pct_change()
        monthly_returns = monthly_returns.iloc[1: , :]
        
        print("Indices monthly returns calculated.\n")
        return monthly_returns
        
    elif(frequency=="y")or(frequency=="a"):
        print("Calculating indices annual returns...\n")
        annual_returns = pd.DataFrame(columns = indices.columns)
        append_data = []
        
        # for loop with an incrementation of 252 => we take the data every year
        for i in range(0, len(indices), 252):
            append_data.append(indices.iloc[[i]])
        annual_returns = pd.concat(append_data)
        annual_returns = annual_returns.pct_change()
        annual_returns = annual_returns.iloc[1: , :]
        
        print("Indices annual returns calculated.\n")
        return annual_returns
        
### Calculate score for each pair of assets with rolling window ###
def score(data, signals, returns, win):
    print("Calculating score for each pair of assets...\n")
    cc = list(combinations(data.columns,2)) #Define all pairs in the portfolio
    df = pd.DataFrame(cc).transpose()
    df = df.rename(index={0: 'Pair_1', 1: 'Pair_2'})
    limit_index = len(returns)-len(signals) 
    d_rets = returns.iloc[limit_index:len(returns)]
    B11 = []
    B22 = []
    ro12 = []
    B21 = []
    B12 = []
    sc = []
    
    #For each asset pair ij, each day we estimate its composite score, θij , using a rolling window of trailing signals and returns of length 100 days
    for i in range(len(signals)-win):
        driver1 = signals[i:i+win].corrwith(d_rets[i:i+win], axis = 0)        
        driver2 = signals[i:i+win].corr()
        
        corr_cross = pd.concat([signals[i:i+win], d_rets[i:i+win]], axis = 1) #col = signal actif i, index = rdt actif j
        driver3 = corr_cross.corr()
        driver3 = driver3.copy()
        driver3 = driver3.iloc[15:30,0:15]
                
        for j in range(len(df.columns)):
            
            #Driver 1 : Own-asset predictability
            s = driver1.loc[(driver1.index == df.iloc[0,j])]
            val = s.values[0]
            B11.append(val)
            s2 = driver1.loc[(driver1.index == df.iloc[1,j])]
            val2 = s2.values[0]
            B22.append(val2)
            
            #Driver 2 : Signal correlation
            s3 = driver2[df.iloc[0,j]].loc[driver2.index == df.iloc[1,j]]
            s3_1 = s3.loc[(s3.index == df.iloc[1,j])]
            val3 = s3_1.values[0]
            ro12.append(val3)
            
            #Driver 3 : Cross-asset predictability
            s4 = driver3[df.iloc[0,j]].loc[driver3.index == df.iloc[1,j]]
            s4_1 = s4.loc[(s4.index == df.iloc[1,j])]
            val4 = s4_1.values[0]
            B21.append(val4)
            s5 = driver3[df.iloc[1,j]].loc[driver3.index == df.iloc[0,j]]
            s5_1 = s5.loc[(s5.index == df.iloc[0,j])]
            val5 = s5_1.values[0]
            B12.append(val5)
            
        tab = pd.DataFrame(list(zip(B11, B22, B21, B12, ro12)),
                          columns = ['B11', 'B22', 'B21', 'B12', 'ro12'])
        tab['Score'] = ((tab['B11']-tab['B12']) + (tab['B22']-tab['B21']))*pow((1-tab['ro12'])/np.pi,0.5)            
        sc.append(tab['Score'])
        
        B11 = []
        B22 = []
        ro12 = []
        B21 = []
        B12 = []
        
        print(str(len(signals)-win-(i+1)) + ' remaining scores to calculate...\n')
    
    name = ''
    names = []
    for k in range(len(df.columns)):
        name = df.iloc[0,k] + " / " + df.iloc[1,k]
        names.append(name)
        
    scores = pd.DataFrame(sc)
    scores.columns = names
    scores = scores.set_index(signals.index[win:len(signals)])
        
    print("Scores calculated.\n")    
    return scores, df

### Determine daily Long-Neutral-Short positions for each asset ###
def CSmom_repartition_LNS(ranking,long,short):
    print("Defining daily LNS repartition...\n")
    repartition_LNS = ranking.copy()

    long_threshold = len(repartition_LNS.columns) - round(long * len(repartition_LNS.columns)) #to define top x best performing assets
    short_threshold = round(long * len(repartition_LNS.columns)) #to define top x worst performing assets
   
    # we get the position 1, 0, -1 for each asset regarding to its ranking
    for col in repartition_LNS.columns:
        repartition_LNS[col] = np.where(repartition_LNS[col] >= long_threshold, 1, np.where(repartition_LNS[col] <= short_threshold, -1, 0 ))

    print("Daily LNS repartition defined.\n")
    return repartition_LNS

### Determine daily Long-Neutral-Short positions for each asset based on pairs score ###
def PAIRSmom_repartition_LNS(ranking, long, pairs, signals):
    print("Defining daily LNS repartition...\n")
    limit_index = len(signals)-len(ranking) 
    new_signals = signals.iloc[limit_index:len(signals)]
        
    nb_pairs = round(long * len(signals.columns)) #number of top pairs selected in portfolio
    
    #Creation of daily LNS positions based on pairs score    
    daily_LNS = pd.DataFrame(np.zeros([len(new_signals),len(new_signals.columns)]))
    daily_LNS.columns = new_signals.columns
    daily_LNS = daily_LNS.set_index(new_signals.index)

    for i in range(len(ranking)):
        count = 0
        for k in range(1,len(ranking.columns)): #check score from best = 1 to worst k = nb_pairs_of_portfolio
            for j in range(len(ranking.columns)):
                       
                if count < nb_pairs:         #stop when we have the adequate number pairs active for each day
                    
                    if ranking.iloc[i,j] == k: 

                        if new_signals[pairs.iloc[0,j]].values[i] > new_signals[pairs.iloc[1,j]].values[i]: #if signal of asset i > signal of asset j 

                            if daily_LNS[pairs.iloc[0,j]].values[i] != 1 and daily_LNS[pairs.iloc[0,j]].values[i] != -1 and daily_LNS[pairs.iloc[1,j]].values[i] != -1 and daily_LNS[pairs.iloc[1,j]].values[i] != 1:
                                daily_LNS[pairs.iloc[0,j]].values[i] = 1 #long position on asset i with greater signal
                                daily_LNS[pairs.iloc[1,j]].values[i] = -1 #short position on asset j with weaker signal
                                count +=1

                            
                        elif new_signals[pairs.iloc[0,j]].values[i] < new_signals[pairs.iloc[1,j]].values[i]: #if signal of asset i < signal of asset j 

                            if daily_LNS[pairs.iloc[0,j]].values[i] != 1 and daily_LNS[pairs.iloc[0,j]].values[i] != -1 and daily_LNS[pairs.iloc[1,j]].values[i] != -1 and daily_LNS[pairs.iloc[1,j]].values[i] != 1:
                                daily_LNS[pairs.iloc[0,j]].values[i] = -1 #short position on asset i with weaker signal
                                daily_LNS[pairs.iloc[1,j]].values[i] = 1 #long position on asset j with greater signal
                                count +=1

    print("Daily LNS repartition defined.\n")
    return daily_LNS

### Definition of the positions taken in the portfolio ###
def rebalancing_assets(repartition,rebal_type, rebal_duree):
    print("Rebalancing positions in the portfolio...\n")
    #Define daily LNS repartitions
    portfolio_positions = repartition.copy()
    #We start trades at d+1
    portfolio_positions = portfolio_positions.iloc[1: , :]

    if(rebal_type=="d") or (rebal_type=="j"):
        ref_value = portfolio_positions.iloc[0]
        for i in range(0, len(portfolio_positions)):
            if(i%(1*rebal_duree) == 0):
                ref_value = portfolio_positions.iloc[i]
            else:
                portfolio_positions.iloc[i]=ref_value

    elif(rebal_type=="w") or (rebal_type=="s"):
        ref_value = portfolio_positions.iloc[0]
        for i in range(0, len(portfolio_positions)):
            if(i%(5*rebal_duree) == 0):
                ref_value = portfolio_positions.iloc[i]
            else:
                portfolio_positions.iloc[i]=ref_value
    
    elif(rebal_type=="m"):
        ref_value = portfolio_positions.iloc[0]
        for i in range(0, len(portfolio_positions)):
            if(i%(21*rebal_duree) == 0):
                ref_value = portfolio_positions.iloc[i]
            else:
                portfolio_positions.iloc[i]=ref_value
    
        
    elif(rebal_type=="y") or (rebal_type=="a"):
        ref_value = portfolio_positions.iloc[0]
        for i in range(0, len(portfolio_positions)):
            if(i%(252*rebal_duree) == 0):
                ref_value = portfolio_positions.iloc[i]
            else:
                portfolio_positions.iloc[i]=ref_value
    
    
    print("Portfolio rebalanced.\n")
    return portfolio_positions

## Performance calculations ###
def portfolio_returns(portfolio_positions, daily_returns):
    print("Calculating performance of the portfolio...\n")

    limit_index = len(daily_returns)-len(portfolio_positions) 
    d_rets = daily_returns.iloc[limit_index:len(daily_returns)]
    
    portfolio_returns = np.multiply(d_rets,portfolio_positions)
    portfolio_returns = portfolio_returns.replace(-0, 0)
    
    portfolio_returns_inpct = portfolio_returns.select_dtypes(include=['number']) * 100
    portfolio_returns_inpct = portfolio_returns_inpct.round(2)
    
    portfolio_daily_performance = round(portfolio_returns_inpct.mean(axis = 1),2)
    
    maxValues = portfolio_returns_inpct.max(axis=1)
    maxValues_asset = portfolio_returns_inpct.idxmax(axis=1)
    
    minValues = portfolio_returns_inpct.min(axis=1)
    minValues_asset = portfolio_returns_inpct.idxmin(axis=1)
    
    dico = {'%Daily returns':portfolio_daily_performance, 'Best performing asset':maxValues_asset, '%Perf_best':maxValues,'Worst performing asset':minValues_asset, '%Perf_worst':minValues}
    perfs = pd.DataFrame(dico)

    print("Performance calculated.\n")
    return perfs, portfolio_returns_inpct

## Performance calculations with rebalancing cost ###
def performance_with_rebalancing_cost(returns,rebal_type, rebal_duree,rebal_cost):
    print("Calculating performance of the portfolio with rebalancing cost...\n")
    ajusted_performance = returns.copy()
    #Convert cost in bps in percent
    rebal_cost_inpct = -rebal_cost * 0.01
    costs = []
    
    if(rebal_type=="d")or(rebal_type=="j"):
        for i in range(0, len(returns)):
            if(i%(1*rebal_duree) == 0):
                costs.append(rebal_cost_inpct)
            else:
                costs.append(np.nan)

    elif(rebal_type=="w")or(rebal_type=="s"):
        for i in range(0, len(returns)):
            if(i%(5*rebal_duree) == 0):
                costs.append(rebal_cost_inpct)
            else:
                costs.append(np.nan)
    
    elif(rebal_type=="m"):
        for i in range(0, len(returns)):
            if(i%(21*rebal_duree) == 0):
                costs.append(rebal_cost_inpct)
            else:
                costs.append(np.nan)
    
        
    elif(rebal_type=="y")or(rebal_type=="a"):
        for i in range(0, len(returns)):
            if(i%(252*rebal_duree) == 0):
                costs.append(rebal_cost_inpct)
            else:
                costs.append(np.nan)
    
    ajusted_performance['Rebal_cost'] = costs
    portfolio_returns_with_cost = round(ajusted_performance.mean(axis = 1),2)
    print("Performance with rebalancing cost calculated.\n")
    return portfolio_returns_with_cost

### Strategy cumulative returns and annualized return ### 
def portfolio_cum_returns(portfolio_daily_returns):
    daily_perf = portfolio_daily_returns.iloc[:, 0]

    cum_returns = pd.DataFrame(np.zeros([len(daily_perf)]))
    cum_returns = cum_returns.set_index(daily_perf.index)

    sum_cum_returns = daily_perf.iloc[1]
     
    for i in range(len(cum_returns)):
        sum_cum_returns += daily_perf.iloc[i]
        cum_returns.iloc[i] = sum_cum_returns 
    
    return cum_returns

def portfolio_annual_return(portfolio_daily_returns):
    daily_perf = portfolio_daily_returns.iloc[:, 0]
    annualized_perf = (daily_perf.sum())/np.sqrt(len(portfolio_daily_returns))
    return annualized_perf

def sharpe_ratio(annualized_performance, std):
    sharpe_interm = annualized_performance/std
    sharpe = np.sqrt(252) * sharpe_interm
    return sharpe

### Main ###
def cleaning_and_splitting(data, repartition_train):
    data_cleaned = clean_dataset(data)
    data_train,data_test = split_train_test(data_cleaned,repartition_train)     
    return data_train, data_test
    
def CSmom(data, mom_type, mom_duree, long, short, rebal_type, rebal_duree, rebal_cost):
    print("Running CSmom strategy...\n")
    signals = momentum_signals(data, mom_type, mom_duree)

    daily_returns = returns(data,"d")

    daily_ranking = signals.rank(axis = 1)

    daily_positions = CSmom_repartition_LNS(daily_ranking, long, short)

    portfolio_positions = rebalancing_assets(daily_positions, rebal_type, rebal_duree)

    portfolio_daily_rets, portfolio_assets_returns = portfolio_returns(portfolio_positions,daily_returns)
    
    #Add performance with rebalancing cost and change between perf with and without cost 
    portfolio_daily_rets.insert(1,"%Daily returns with cost",performance_with_rebalancing_cost(portfolio_assets_returns, rebal_type, rebal_duree, rebal_cost))    
    portfolio_daily_rets.insert(2,"%Change",portfolio_daily_rets["%Daily returns with cost"].subtract(portfolio_daily_rets["%Daily returns"]))
    
    print("CSmom strategy finished.\n")
    return portfolio_daily_rets

def PAIRSmom(data, mom_type, mom_duree, long, window, rebal_type, rebal_duree, rebal_cost):
    print("Running PAIRSmom strategy...\n")
    signals = momentum_signals(data, mom_type, mom_duree)

    daily_returns = returns(data,"d")

    scores, pairs = score(data, signals, daily_returns, window)

    daily_ranking = scores.rank(axis = 1)
     
    daily_positions = PAIRSmom_repartition_LNS(daily_ranking, long, pairs, signals)

    portfolio_positions = rebalancing_assets(daily_positions, rebal_type, rebal_duree)

    portfolio_daily_rets, portfolio_assets_returns = portfolio_returns(portfolio_positions,daily_returns)
    
    #Add performance with rebalancing cost and change between perf with and without cost + cumulative returns
    portfolio_daily_rets.insert(1,"%Daily returns with cost",performance_with_rebalancing_cost(portfolio_assets_returns, rebal_type, rebal_duree, rebal_cost))    
    portfolio_daily_rets.insert(2,"%Change",portfolio_daily_rets["%Daily returns with cost"].subtract(portfolio_daily_rets["%Daily returns"]))
    
    print("PAIRSmom strategy finished.\n")
    return portfolio_daily_rets

data_train, data_test = cleaning_and_splitting(all_data, inputs_data.iloc[0,0])

#CSmom variables
CSmom_returns = CSmom(data_train, inputs_data.iloc[0,2],inputs_data.iloc[0,1],inputs_data.iloc[0,3],inputs_data.iloc[0,4], inputs_data.iloc[0,6],inputs_data.iloc[0,5],inputs_data.iloc[0,7])
CSmom_cum_returns = portfolio_cum_returns(CSmom_returns)
CSmom_annual_return = portfolio_annual_return(CSmom_returns)
CSmom_std_returns = CSmom_returns.iloc[:, 0].std()*np.sqrt(len(CSmom_returns))
CSmom_sharpe = sharpe_ratio(CSmom_annual_return, CSmom_std_returns)

#Pairsmom variables
PAIRSmom_returns = PAIRSmom(data_train, inputs_data.iloc[0,2], inputs_data.iloc[0,1], inputs_data.iloc[0,3], inputs_data.iloc[0,8], inputs_data.iloc[0,6],inputs_data.iloc[0,5],inputs_data.iloc[0,7])
PAIRSmom_cum_returns = portfolio_cum_returns(PAIRSmom_returns)
PAIRSmom_annual_return = portfolio_annual_return(PAIRSmom_returns)
PAIRSmom_std_returns = PAIRSmom_returns.iloc[:, 0].std()*np.sqrt(len(PAIRSmom_returns))
PAIRSmom_sharpe = sharpe_ratio(PAIRSmom_annual_return, PAIRSmom_std_returns)

### Plots ###

ax = CSmom_returns.iloc[:, 0].plot()
PAIRSmom_returns.iloc[:, 0].plot(ax=ax)
ax.legend(['Benchmark', 'Pairs'])
plt.title("Daily performance without rebalancing cost")
plt.show()

ax = CSmom_returns.iloc[:, 1].plot()
PAIRSmom_returns.iloc[:, 1].plot(ax=ax)
ax.legend(['Benchmark', 'Pairs'])
plt.title("Daily performance with rebalancing cost")
plt.show()

ax = CSmom_cum_returns.plot()
PAIRSmom_cum_returns.plot(ax=ax)
ax.legend(['Benchmark', 'Pairs'])
plt.title("Cumulative returns")
plt.show()

print('\n-------------------------------------------------------------------------------------------\n')      
print('Parameters:\n')
print('Period: ' + str(round(len(data_train)/252)) +' years \n')
print('Start date: ' + str(data_train.index[0]) +'\n')
print('End date: ' + str(CSmom_returns.index[len(CSmom_returns)-1]) +'\n')
print('Momentum used to calculate signals: ' + str(inputs_data.iloc[0,1]) + ' ' + str(inputs_data.iloc[0,2]+'\n'))
print('Rebalancing frequency: ' + str(inputs_data.iloc[0,5]) + ' ' + str(inputs_data.iloc[0,6])+'\n')
print('Benchmark strategy: Long repartition: ' + str(inputs_data.iloc[0,3]) +'\n')
print('Benchmark strategy: Short repartition: ' + str(inputs_data.iloc[0,4]) +'\n')
print('Pairs strategy: Number of pairs: ' + str(round(inputs_data.iloc[0,3] * len(data_train.columns))) +'\n')


print('-------------------------------------------------------------------------------------------\n')      
print('Statistics:\n')
print('Benchmark strategy: Annualized return (%) : ' + str(round(CSmom_annual_return,2)) +'\n')
print('Pairs strategy: Annualized return (%) : ' + str(round(PAIRSmom_annual_return,2)) +'\n')
print('Benchmark strategy: Standard deviation: ' + str(round(CSmom_std_returns,2)) +'\n')
print('Pairs strategy: Standard deviation: ' + str(round(PAIRSmom_std_returns,2)) +'\n')
print('Benchmark strategy: Sharpe ratio: ' + str(round(CSmom_sharpe,2)) +'\n')
print('Pairs strategy: Sharpe ratio: ' + str(round(PAIRSmom_sharpe,2)) +'\n')
print('Benchmark strategy: Cumulative return (%): ' + str(round(CSmom_cum_returns.iloc[len(CSmom_cum_returns)-1,0],2)) +'\n')
print('Pairs strategy: Cumulative return (%): ' + str(round(PAIRSmom_cum_returns.iloc[len(PAIRSmom_cum_returns)-1,0],2)) +'\n')
print('-------------------------------------------------------------------------------------------\n') 

