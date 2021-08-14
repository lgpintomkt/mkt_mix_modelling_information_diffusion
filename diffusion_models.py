# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 21:53:23 2021

@author: MMSETUBAL
"""
from scipy.optimize import least_squares
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data_path="C:\\Users\\MMSETUBAL\\Desktop\\Artigos\\MMM and Information Diffusion over Complex Networks"
data=pd.read_csv(data_path+"\\Data\\Statistical\\mmm_data.csv",sep=";").set_index("t")

#Market Potential - Mobile Broadband Subscriptions (Japan Dec. 2019)
m=data['Adoption'].iloc[-1]
adoption=np.array(data['Adoption'])
t=np.array(data.index)
mmix=np.array(data[['Product','Price','Place','Promotion']])

#Bass Diffusion (BD)
def bass_model(t,p,q,m):
    sales=0
    adoption=[]
    for i in t:
        sales+=p*m+(q-p)*sales-(q/m)*(sales**2)
        adoption.append(sales)
    return adoption

def bass_residuals(x,f,t,m):
    p=x[0]
    q=x[1]
    return bass_model(t,p,q,m)-f

bass_diffusion=least_squares(bass_residuals, x0=[0.03,0.38], args=(adoption,t,m))

p=bass_diffusion.x[0]
q=bass_diffusion.x[1]
bass_forecast=bass_model(t,p,q,m)

#Mesak Diffusion (MD)
def mesak_model(t,p,q,m,lam,mmix):
    price=mmix[:,1]**(-lam)
    place=mmix[:,2]
    promotion=np.sqrt(mmix[:,3])
    sales=0
    adoption=[]
    for i in t:
        sales+=(p*price[i-1]+q*sales)*(m*place[i-1]-sales)*promotion[i-1]
        adoption.append(sales)
    return adoption

def mesak_residuals(x,f,t,m,mmix):
    p=x[0]
    q=x[1]
    lam=x[2]
    return np.sum((mesak_model(t,p,q,m,lam,mmix)-f)**2)

mesak_diffusion=minimize(mesak_residuals, method='nelder-mead', tol=1e-6, x0=[1e-12,1e-12,-1], args=(adoption,t,m,mmix))

p=mesak_diffusion.x[0]
q=mesak_diffusion.x[1]
lam=mesak_diffusion.x[2]
mesak_forecast=mesak_model(t,p,q,m,lam,mmix)

#General Marketing Mix Modelling (GMM)

#Additive GMM (AGMM)

#Additive GMM over Complex Networks (AGMM-CN)

forecasts=pd.DataFrame({
    'Observed':adoption,
    'Bass Diffusion':bass_forecast,
    'Mesak Diffusion':mesak_forecast
    },index=t).to_excel(data_path+"\\results.xlsx")
