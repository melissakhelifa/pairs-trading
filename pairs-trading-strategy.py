# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 17:31:46 2021

@author: Mélissa
"""
import pandas as pd
import numpy as np

date_limite = '03/12/2010'

all_data = pd.read_csv ('.\data\data_mef.csv',delimiter=';')
all_data['Date'] = pd.to_datetime(all_data['Date'],dayfirst=True) 
    #print(all_data.dtypes)

test_data_raw = all_data.loc[all_data['Date'] <= date_limite]
future_data_raw = all_data.loc[all_data['Date'] > date_limite]

    # ffill: propagate last valid observation forward to next valid backfill
    # bfill: use next valid observation to fill gap

test_data = test_data_raw.fillna(method="ffill")
test_data = test_data.set_index('Date')

### Calculate Momentum Signal ###
def calcul_momentum_signal(dataset,mom_duree,mom_type):
    momentum_signal = dataset.copy()
    nb_bdays_per_month = 22
    nb_bdays_per_year = nb_bdays_per_month*12
    if mom_type == "d" or mom_type == "j":
        momentum_signal = momentum_signal.apply(func = lambda x: x.shift(mom_duree)-x.shift(1), axis = 0)
        return momentum_signal
    elif mom_type == "m":
        momentum_signal = momentum_signal.apply(func = lambda x: x.shift(mom_duree*nb_bdays_per_month)-x.shift(1), axis = 0)
        return momentum_signal
    elif mom_type == "y"or mom_type == "a":
        momentum_signal = momentum_signal.apply(func = lambda x: x.shift(mom_duree*nb_bdays_per_year)-x.shift(1), axis = 0)
        return momentum_signal
   
### Calculate Returns ###
def calcul_rend_signal(momentum_signals):
    returns_data = momentum_signals.copy()
    returns_data = returns_data.apply(func = lambda x: x.shift(-1)/x - 1, axis = 0)
    return returns_data

### Determine Long-Neutral-Short positions for each asset ###
def ponderation_actifs(momentum_signals,long,short):
    repartition_LNS = momentum_signals.rank(axis = 1)
    for col in repartition_LNS.columns:
        repartition_LNS[col] = np.where(repartition_LNS[col] >= round(long*len(repartition_LNS.columns)), 1, np.where(repartition_LNS[col] <= round(short*len(repartition_LNS.columns)), -1, 0))
    return repartition_LNS

### Test ###
signals = calcul_momentum_signal(test_data,11,"m")
return_signals = calcul_rend_signal(signals)
repartition_LNS = ponderation_actifs(signals,1/3,1/3)
portfolio = np.multiply(return_signals,repartition_LNS)
portfolio_returns = portfolio.sum(axis=1)/15






    




    


    