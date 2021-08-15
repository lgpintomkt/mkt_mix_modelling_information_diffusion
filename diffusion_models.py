# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 21:53:23 2021

@author: MMSETUBAL
"""
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import inteq
import math
from math import isnan,isinf

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
def mesak_model(t,p,q,m,mmix,w0,w1,w2,w3,eps=1e-2):
    price=w1*mmix[:,1]+eps
    place=w2*mmix[:,2]+eps
    promotion=w3*np.sqrt(mmix[:,3])+eps
    sales=0
    adoption=[]
    for i in t:
        sales+=w0+(p*price[i-1]+q*sales)*(m*place[i-1]-sales)*promotion[i-1]
        adoption.append(sales)
    return adoption

def mesak_residuals(x,f,t,m,mmix):
    p=x[0]
    q=x[1]
    w0=x[2]
    w1=x[3]
    w2=x[4]
    w3=x[5]
    return np.sum((mesak_model(t,p,q,m,mmix,w0,w1,w2,w3)-f)**2)+np.sum(np.abs(x))

mesak_diffusion=minimize(mesak_residuals, method='nelder-mead', x0=[1e-12,1e-12,1e8,-1,1,1], args=(adoption,t,m,mmix))

p=mesak_diffusion.x[0]
q=mesak_diffusion.x[1]
w0=mesak_diffusion.x[2]
w1=mesak_diffusion.x[3]
w2=mesak_diffusion.x[4]
w3=mesak_diffusion.x[5]
mesak_forecast=mesak_model(t,p,q,m,mmix,w0,w1,w2,w3)

#General Marketing Mix Modelling (GMM)
def gmm_model(t,p,q,m,mmix,w0,w1,w2,w3,w4,w5,eps=1e-2):
    def kernel(T,xi):
        tf=int(max(np.ceil(T))[0])
        prod=(T*tf)[:,0]
        price=w1*mmix[:,1]+eps
        place=w2*mmix[:,2]+eps
        promotion=w3*np.sqrt(mmix[:,3])+eps
        sales=0
        adoption=[]
        for i in xi:
            ti=int(math.floor(i))
            tf=int(math.ceil(i))
            if ti == mmix.shape[0]:
                ti = mmix.shape[0]-1
            if tf == mmix.shape[0]:
                tf = mmix.shape[0]-1    
            pric=(price[ti]+price[tf])/2
            plac=(place[ti]+place[tf])/2
            promo=(promotion[ti]+promotion[tf])/2
            try:
                last_sales=adoption[-1]
            except:
                last_sales=0
            sales+=w0+(p*pric+q*last_sales)*(m*plac-sales)*promo
            adoption.append(sales)
        return prod*np.array(adoption)
                
    def g(xi):
        return np.exp(xi/w5)
    
    price=w1*mmix[:,1]+eps
    place=w2*mmix[:,2]+eps
    promotion=w3*np.sqrt(mmix[:,3])+eps
    sales=0
    adoption=[]
    
    for i in t:
        volterra_matrix=inteq.SolveVolterra(kernel,g,a=0,b=i)
        volterra_dict=dict(zip(volterra_matrix[0,:], volterra_matrix[1,:].T))
        volterra_dict={k: volterra_dict[k] for k in volterra_dict if (not isnan(volterra_dict[k])) and (not isinf(volterra_dict[k]))}
        volterra_solution=volterra_dict.get(i, volterra_dict[min(volterra_dict.keys(), key=lambda k: abs(k-i))])  
        
        product=(math.exp(-i/w5)*volterra_solution+w4*w5*math.exp(-i/w5))
        sales+=w0+(p*price[i-1]+q*product)*(m*place[i-1]-sales)*promotion[i-1]
        adoption.append(sales)
    return adoption

def gmm_residuals(x,f,t,m,mmix):
    p=x[0]
    q=x[1]
    w0=x[2]
    w1=x[3]
    w2=x[4]
    w3=x[5]
    w4=x[6]
    w5=x[7]
    return np.sum((gmm_model(t,p,q,m,mmix,w0,w1,w2,w3,w4,w5)-f)**2)+np.sum(np.abs(x))

gmm_diffusion=minimize(gmm_residuals, method='L-BFGS-B',options={'maxiter':100}, x0=[1e-12,1e-12,1e8,-1,1,1,1e-5,1e-3], args=(adoption,t,m,mmix))

p=gmm_diffusion.x[0]
q=gmm_diffusion.x[1]
w0=gmm_diffusion.x[2]
w1=gmm_diffusion.x[3]
w2=gmm_diffusion.x[4]
w3=gmm_diffusion.x[5]
w4=gmm_diffusion.x[6]
w5=gmm_diffusion.x[7]
gmm_forecast=gmm_model(t,p,q,m,mmix,w0,w1,w2,w3,w4,w5)

#Additive GMM (AGMM)

#Additive GMM over Complex Networks (AGMM-CN)

forecasts=pd.DataFrame({
    'Observed':adoption,
    'Bass Diffusion':bass_forecast,
    'Mesak Diffusion':mesak_forecast,
    'General Marketing Mix Model':gmm_forecast
    },index=t)

forecasts.to_excel(data_path+"\\results.xlsx")
