# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 17:02:01 2021

@author: MMSETUBAL
"""
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt

#Place
data_path="C:\\Users\\MMSETUBAL\\Desktop\\Artigos\\MMM and Information Diffusion over Complex Networks"

endog=pd.read_csv(data_path+"\\Data\\Statistical\\digital_distribution_data.csv",sep=";").set_index("Date")

fig, ax = plt.subplots(figsize=(13,3))
ax.plot(endog['nttdocomo.co.jp'].diff(1))

fig, ax = plt.subplots(figsize=(13,3))
ax.plot(endog['KDDI.com'].diff(1))

fig, ax = plt.subplots(figsize=(13,3))
ax.plot(endog['softbank.jp'].diff(1))



mod = sm.tsa.DynamicFactor(endog, k_factors=1, factor_order=1, error_order=1)
initial_res = mod.fit(method='powell', disp=False)
res = mod.fit(initial_res.params, disp=False)

dates = endog.index._mpl_repr()
fig, ax = plt.subplots(figsize=(13,3))
ax.plot(dates, res.factors.filtered[0], label='Factor')
ax.legend()

pd.DataFrame(res.factors.filtered[0]).to_excel(data_path+"\\Data\\Statistical\\digital_distribution_factor.xlsx")
