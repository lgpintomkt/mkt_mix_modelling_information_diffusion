# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 21:53:23 2021

@author: MMSETUBAL
"""
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.optimize import approx_fprime
import pandas as pd
import numpy as np
import inteq
import math
import requests
from math import isinf
#import sgd-for-scipy from github gist
exec(requests.get('https://gist.githubusercontent.com/lgpintomkt/9dbf22eb514d275cd89be1172477a1e8/raw/2898c5c73db13936baa7bb673a73f485bb151a66/sgd-for-scipy.py').text)

data_path="C:\\Users\\MMSETUBAL\\Desktop\\Artigos\\MMM and Information Diffusion over Complex Networks"
data=pd.read_csv(data_path+"\\Data\\Statistical\\mmm_data.csv",sep=";").set_index("t")

#Market Potential - Mobile Broadband Subscriptions (Japan Dec. 2019)
m=data['Adoption'].iloc[-1]
adoption=np.array(data['Adoption'])
t=np.array(data.index)
t_train=np.array(data.index)[:60]
t_test=np.array(data.index)[60:]
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

bass_diffusion=least_squares(
    bass_residuals, 
    x0=[0.03,0.38], 
    args=(adoption,t,m)
    )

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

mesak_diffusion=minimize(
    mesak_residuals, 
    method='nelder-mead', 
    x0=[1e-12,1e-12,1e8,-1,1,1], 
    args=(adoption,t,m,mmix)
    )

p=mesak_diffusion.x[0]
q=mesak_diffusion.x[1]
w0=mesak_diffusion.x[2]
w1=mesak_diffusion.x[3]
w2=mesak_diffusion.x[4]
w3=mesak_diffusion.x[5]
mesak_forecast=mesak_model(t,p,q,m,mmix,w0,w1,w2,w3)

#General Marketing Mix Modelling (GMM)
def gmm_model(t,p,q,m,mmix,w0,w1,w2,w3,w4,w5,eps=1e-2):  
    price=w1*mmix[:,1]+eps
    place=w2*mmix[:,2]+eps
    promotion=w3*np.sqrt(mmix[:,3])+eps
    sales=0
    adoption=[]
    volterra_dict=quality_integral(t,p,q,m,mmix,w0,w1,w2,w3,w4,w5,eps)

    for i in t:
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
    penalty=0
    # if np.abs(w0) > 2e8:
    #     penalty+=w0**2
    # if np.abs(p) > 1e3:
    #     penalty+=p**2
    return np.sum((gmm_model(t,p,q,m,mmix,w0,w1,w2,w3,w4,w5)-f)**2)+np.sum(np.abs(x))+penalty

def quality_integral(t,p,q,m,mmix,w0,w1,w2,w3,w4,w5,eps=1e-2):
    def kernel(t,xi):
        product=mmix[:,0]+eps
        price=w1*mmix[:,1]+eps
        place=w2*mmix[:,2]+eps
        promotion=w3*np.sqrt(mmix[:,3])+eps
        sales=0
        adoption=[] 
        for i in xi:
        # for ti, tf in zip(xi, xi[1:]):
        #     upper=math.ceil(tf)
        #     lower=math.floor(tf)
            
        #     if upper == len(mmix[:,0]):
        #         upper=len(mmix[:,0])-1
        #     if lower == len(mmix[:,0]):
        #         lower=len(mmix[:,0])-1
                
        #     diff=tf-ti
        
            #Exponential and Polynomial Interpolation
            #Product - See spreadsheet "Product" in https://docs.google.com/spreadsheets/d/1pbBUsyReOX6y5hdoTWQkeU3hxhrW_7WOdALjrJT1y7I/edit?usp=sharing
            a=5.195836994
            b=0.02509865638
            _prod=a*math.exp(b*i)
            
            #Perceived Quality transformation using Weber-Fechner Law which linearizes product quality
            min_quality_threshold=0.01 #mbps
            weber_fechner_constant=1
            prod=weber_fechner_constant*math.log(_prod/min_quality_threshold)
            
            #Price - See spreadsheet "Price" in https://docs.google.com/spreadsheets/d/1pbBUsyReOX6y5hdoTWQkeU3hxhrW_7WOdALjrJT1y7I/edit?usp=sharing
            p0=0.7238312454
            p1=-0.009686677157
            p2=0.0001198351799
            p3=-0.00001543875789
            p4=1.30E-07
            pric=p0+p1*i+p2*(i**2)+p3*(i**3)+p4*(i**4)
            
            #Place - See spreadsheet "Distribution" in https://docs.google.com/spreadsheets/d/1pbBUsyReOX6y5hdoTWQkeU3hxhrW_7WOdALjrJT1y7I/edit?usp=sharing
            p0=-4.590334248
            p1=-0.009876083408
            p2=0.000929745442
            p3=-0.00004219246514
            p4=3.48E-07
            plac=p0+p1*i+p2*(i**2)+p3*(i**3)+p4*(i**4)

            #Promotion - See spreadsheet "Advertising" in https://docs.google.com/spreadsheets/d/1pbBUsyReOX6y5hdoTWQkeU3hxhrW_7WOdALjrJT1y7I/edit?usp=sharing
            p0=-2.90E-04
            p1=5.48E-02
            p2=-3.13E+00
            p3=1.37E+02
            p4=2.06E+04         
            promo=p0+p1*i+p2*(i**2)+p3*(i**3)+p4*(i**4)

            
            #Linear Interpolation
            # if product[lower]>product[upper]:
            #     prod=product[upper]-product[lower]*diff
            # else:
            #     prod=product[lower]+product[upper]*diff
            # if price[lower]>price[upper]:
            #     pric=price[upper]-price[lower]*diff
            # else:
            #     pric=price[lower]+price[upper]*diff
            # if place[lower]>place[upper]:
            #     plac=place[upper]-place[lower]*diff
            # else:
            #     plac=place[lower]+place[upper]*diff
            # if promotion[lower]>promotion[upper]:
            #     promo=promotion[upper]-promotion[lower]*diff
            # else:
            #     promo=promotion[lower]+promotion[upper]*diff
            
            if len(adoption) > 1:
                prev_sales = adoption[-1]
            else:
                prev_sales = 0

            sales+=w0+(p*pric+q*prev_sales)*(m*plac-sales)*promo
            # if isinf(sales):
            #     print("infinity detected")
            adoption.append(sales*prod)
        # sales+=w0+(p*pric+q*prev_sales)*(m*plac-sales)*promo
        # adoption.append(sales*prod)
        return np.array(adoption)
                
    def g(xi):
        return np.exp(xi)/w5
    
    volterra_matrix=inteq.SolveVolterra(kernel,g,a=0,b=max(t))
    volterra_dict=dict(zip(volterra_matrix[0,:], volterra_matrix[1,:].T))
    return volterra_dict
    
    

gmm_diffusion=minimize(
    gmm_residuals, 
    method='TNC',
    jac=lambda x,f,t,m,mmix: approx_fprime(x, gmm_residuals, 1e-12,f,t,m,mmix), 
    x0=[1e-12,1e-12,1e8,-1,1,1,1e-12,1e-12], 
    args=(adoption,t,m,mmix),
    bounds=(
        (-1e5,1e5),
        (-1e5,1e5),
        (0,1e9),
        (-1,1),
        (-1,1),
        (-1,1),
        (-1,1),
        (-1,1)
        )
    )

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
def agmm_model(t,p,q,m,mmix,w0,w1,w2,w3,w4,w5,eps=1e-2):  
    price=w1*mmix[:,1]+eps
    place=w2*mmix[:,2]+eps
    promotion=w3*np.sqrt(mmix[:,3])+eps
    sales=0
    adoption=[]
    volterra_dict=quality_integral(t,p,q,m,mmix,w0,w1,w2,w3,w4,w5,eps)

    for i in t:
        volterra_solution=volterra_dict.get(i, volterra_dict[min(volterra_dict.keys(), key=lambda k: abs(k-i))]) 
        product=(math.exp(-i/w5)*volterra_solution+w4*w5*math.exp(-i/w5))
        radiation=p*price[i-1]*(m*place[i-1]-sales)*promotion[i-1]
        diffusion=q*product
        sales+=radiation+diffusion
        adoption.append(sales)
    return adoption

def agmm_residuals(x,f,t,m,mmix):
    p=x[0]
    q=x[1]
    w0=x[2]
    w1=x[3]
    w2=x[4]
    w3=x[5]
    w4=x[6]
    w5=x[7]
    return np.sum((agmm_model(t,p,q,m,mmix,w0,w1,w2,w3,w4,w5)-f)**2)+np.sum(np.abs(x))

agmm_diffusion=minimize(
    agmm_residuals, 
    method='TNC',
    jac=lambda x,f,t,m,mmix: approx_fprime(x, gmm_residuals, 1e-12,f,t,m,mmix), 
    x0=[1e-12,1e-12,1e8,-1,1,1,1e-12,1e-12], 
    args=(adoption,t,m,mmix),
    bounds=(
        (-1e5,1e5),
        (-1e5,1e5),
        (0,1e10),
        (-1,1),
        (-1,1),
        (-1,1),
        (-1,1),
        (-1,1)
        )
    )

p=agmm_diffusion.x[0]
q=agmm_diffusion.x[1]
w0=agmm_diffusion.x[2]
w1=agmm_diffusion.x[3]
w2=agmm_diffusion.x[4]
w3=agmm_diffusion.x[5]
w4=agmm_diffusion.x[6]
w5=agmm_diffusion.x[7]
agmm_forecast=agmm_model(t,p,q,m,mmix,w0,w1,w2,w3,w4,w5)

#Additive GMM over Complex Networks (AGMM-CN)

forecasts=pd.DataFrame({
    'Observed':adoption,
    'Bass Diffusion':bass_forecast,
    'Mesak Diffusion':mesak_forecast,
    'General Marketing Mix Model':gmm_forecast,
    'Additive General Marketing Mix Model':agmm_forecast
    },index=t)

forecasts.to_excel(data_path+"\\results.xlsx")
