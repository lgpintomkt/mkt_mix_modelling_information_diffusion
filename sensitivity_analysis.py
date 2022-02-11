# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 12:00:54 2022

@author: MMSETUBAL
"""
import scipy.integrate as integrate
from scipy.optimize import least_squares
from scipy.optimize import minimize, differential_evolution
from scipy.optimize import approx_fprime
from scipy.optimize import fmin
from scipy.optimize import minimize_scalar
import math
import pandas as pd
import numpy as np
import inteq
import requests
import networkx as nx
from numba import jit
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")

#import sgd-for-scipy from github gist
exec(requests.get('https://gist.githubusercontent.com/lgpintomkt/9dbf22eb514d275cd89be1172477a1e8/raw/362a8c7f49d4c622af858202b6be0b3f7e33aca3/sgd-for-scipy.py').text)

def diff(li1, li2):
    return list(set(li1) - set(li2))

@jit(nopython=True)
def u_i(i):
    #Exponential Interpolation
    #Product - See spreadsheet "Product" in https://docs.google.com/spreadsheets/d/1pbBUsyReOX6y5hdoTWQkeU3hxhrW_7WOdALjrJT1y7I/edit?usp=sharing
    a=5.195836994
    b=0.02509865638
    _prod=a*np.exp(b*i)
    
    #Perceived Quality transformation using Weber-Fechner Law which linearizes product quality
    min_quality_threshold=0.01 #mbps
    weber_fechner_constant=1
    __prod=weber_fechner_constant*np.log(_prod/min_quality_threshold)
    #Normalization and Weighting
    min_prod=6.278126569
    max_prod=8.060131172

    return np.divide((__prod-min_prod),(max_prod-min_prod))

@jit(nopython=True)
def f(i:float,u:float,bc:float,p:float,q:float,m:int,w0:float,w1:float,w2:float,w3:float,w4:float,w5:float):

        #Polynomial Interpolation
            
        #Price - See spreadsheet "Price" in https://docs.google.com/spreadsheets/d/1pbBUsyReOX6y5hdoTWQkeU3hxhrW_7WOdALjrJT1y7I/edit?usp=sharing
        p0=0.7238312454
        p1=-0.009686677157
        p2=0.0001198351799
        p3=-0.00001543875789
        p4=1.30E-07
        __pric=p0+p1*i+p2*(i**2)+p3*(i**3)+p4*(i**4)
        #Normalization
        min_pric=-1.6207472
        max_pric=0.7142491
        pric=w1*(__pric-min_pric)/(max_pric-min_pric)
        
        #Place - See spreadsheet "Distribution" in https://docs.google.com/spreadsheets/d/1pbBUsyReOX6y5hdoTWQkeU3hxhrW_7WOdALjrJT1y7I/edit?usp=sharing
        p0=-4.590334248
        p1=-0.009876083408
        p2=0.000929745442
        p3=-0.00004219246514
        p4=3.48E-07
        __plac=p0+p1*i+p2*(i**2)+p3*(i**3)+p4*(i**4)
        #Normalization
        min_plac=-6.88E+00
        max_plac=-4.60E+00
        plac=w2*(__plac-min_plac)/(max_plac-min_plac)

        #Promotion - See spreadsheet "Advertising" in https://docs.google.com/spreadsheets/d/1pbBUsyReOX6y5hdoTWQkeU3hxhrW_7WOdALjrJT1y7I/edit?usp=sharing
        p0=-2.90E-04
        p1=5.48E-02
        p2=-3.13E+00
        p3=1.37E+02
        p4=2.06E+04         
        __promo=p0+p1*i+p2*(i**2)+p3*(i**3)+p4*(i**4)
        #Normalization
        min_promo=20733.97
        max_promo=26908.49
        promo=w3*(__promo-min_promo)/(max_promo-min_promo)

        return bc+w0+(p*pric+q*u)*(m*plac-bc)*promo
    
def _f(i:float,u:float,bc:float,p:float,q:float,m:int,w0:float,w1:float,w2:float,w3:float,w4:float,w5:float):

        #Polynomial Interpolation
            
        #Price - See spreadsheet "Price" in https://docs.google.com/spreadsheets/d/1pbBUsyReOX6y5hdoTWQkeU3hxhrW_7WOdALjrJT1y7I/edit?usp=sharing
        p0=0.7238312454
        p1=-0.009686677157
        p2=0.0001198351799
        p3=-0.00001543875789
        p4=1.30E-07
        __pric=p0+p1*i+p2*(i**2)+p3*(i**3)+p4*(i**4)
        #Normalization
        min_pric=-1.6207472
        max_pric=0.7142491
        pric=w1*(__pric-min_pric)/(max_pric-min_pric)
        
        #Place - See spreadsheet "Distribution" in https://docs.google.com/spreadsheets/d/1pbBUsyReOX6y5hdoTWQkeU3hxhrW_7WOdALjrJT1y7I/edit?usp=sharing
        p0=-4.590334248
        p1=-0.009876083408
        p2=0.000929745442
        p3=-0.00004219246514
        p4=3.48E-07
        __plac=p0+p1*i+p2*(i**2)+p3*(i**3)+p4*(i**4)
        #Normalization
        min_plac=-6.88E+00
        max_plac=-4.60E+00
        plac=w2*(__plac-min_plac)/(max_plac-min_plac)

        #Promotion - See spreadsheet "Advertising" in https://docs.google.com/spreadsheets/d/1pbBUsyReOX6y5hdoTWQkeU3hxhrW_7WOdALjrJT1y7I/edit?usp=sharing
        p0=-2.90E-04
        p1=5.48E-02
        p2=-3.13E+00
        p3=1.37E+02
        p4=2.06E+04         
        __promo=p0+p1*i+p2*(i**2)+p3*(i**3)+p4*(i**4)
        #Normalization
        min_promo=20733.97
        max_promo=26908.49
        promo=w3*(__promo-min_promo)/(max_promo-min_promo)

        return tf.cast(bc,tf.float32)+tf.cast(w0,tf.float32)+tf.math.multiply(
                tf.math.multiply(
                    tf.math.add(
                        tf.math.multiply(
                            tf.cast(p,tf.float32),
                            tf.cast(pric,tf.float32)),
                        tf.math.multiply(
                            tf.cast(q,tf.float32),
                            tf.cast(u,tf.float32))),
                    tf.math.add(
                        tf.math.multiply(
                            tf.cast(m,tf.float32),
                            tf.cast(plac,tf.float32)),
                        -tf.cast(bc,tf.float32))),
                    tf.cast(promo,tf.float32))

def _u_i(i):
    #Exponential Interpolation
    #Product - See spreadsheet "Product" in https://docs.google.com/spreadsheets/d/1pbBUsyReOX6y5hdoTWQkeU3hxhrW_7WOdALjrJT1y7I/edit?usp=sharing
    a=5.195836994
    b=0.02509865638
    _prod=a*tf.math.exp(b*i)
    
    #Perceived Quality transformation using Weber-Fechner Law which linearizes product quality
    min_quality_threshold=0.01 #mbps
    weber_fechner_constant=1
    __prod=weber_fechner_constant*tf.math.log(_prod/min_quality_threshold)
    #Normalization and Weighting
    min_prod=6.278126569
    max_prod=8.060131172

    return tf.math.divide((__prod-min_prod),(max_prod-min_prod))


solveVolterraJIT=jit(inteq.SolveVolterra)
integrate_quadJIT=jit(integrate.quad)

data_path="C:\\Users\\MMSETUBAL\\Desktop\\Artigos\\MMM and Information Diffusion over Complex Networks"
data=pd.read_csv(data_path+"\\Data\\Statistical\\mmm_data.csv",sep=";").set_index("t")
G=nx.read_graphml(data_path+"\\Data\\Network\\Network Files\\japan_municipalities_extended_normalized_v2.xml")

# print("Normalizing edge weights...")
# for index,(node_i,node_j) in enumerate(list(itertools.product(G.nodes(),G.nodes()))):
#     print(index)
#     normalized=nx.algorithms.structuralholes.normalized_mutual_weight(G,node_i,node_j,weight='weight')
#     nx.set_edge_attributes(G, {(node_i, node_j): {"weight": normalized}})
# print("Done")

#nx.write_graphml(G,data_path+'\\Network\\Network Files\\japan_municipalities_extended_normalized.xml')

#Market Potential - Mobile Broadband Subscriptions (Japan Dec. 2019)
#m=data['Adoption'].iloc[-1]
#Boundary condition: cumulative sales on t=1
bc=11197800 #cumulative subscriptions in Dec2013 - see https://data.oecd.org/broadband/mobile-broadband-subscriptions.htm
adoption=np.array(data['Adoption'])
t=np.array(data.index)
t_train=np.array(data.index)[:60]
t_test=np.array(data.index)[60:]
mmix=np.array(data[['Product','Price','Place','Promotion']])

#Sigma (Std. Dev.) will be 10% of base adoption level
sigma=1125527

#Random Walk Generation
#Adoptions
random_walks=[]
for iteration in range(10000):
    random_walk=[]
    for monthly_adoption in adoption:
        random_walk.append(monthly_adoption+np.random.normal(0,sigma))
    random_walks.append(random_walk)
    
#Marketing Mix Variables
rws_mmix=[]
for iteration in range(10000):
    rw_mmix=np.copy(mmix)
    for (x,y), value in np.ndenumerate(mmix):
        rw_mmix[x,y]=min(1,max(0,value+np.random.normal(0,0.1)))
    rws_mmix.append(rw_mmix)
        
# #Bass Diffusion (BD)
def bass_model(t,p,q,m,bc):
    sales=bc
    adoption=[]
    for i in t:
        sales+=p*m+(q-p)*sales-(q/m)*(sales**2)
        adoption.append(max(0,sales))
    return adoption

def bass_residuals(x,f,t,bc):
    p=x[0]
    q=x[1]
    m=x[2]
    return bass_model(t,p,q,m,bc)-f

# #Mesak Diffusion (MD)
def mesak_model(t,p,q,m,mmix,bc,w0,w1,w2,w3,eps=1e-2):
    price=w1*mmix[:,1]+eps
    place=w2*mmix[:,2]+eps
    promotion=w3*np.sqrt(mmix[:,3])+eps
    sales=bc
    adoption=[]
    for i in t:
        sales+=w0+(p*price[i-1]+q*sales)*(m*place[i-1]-sales)*promotion[i-1]
        adoption.append(max(0,sales))
    return adoption

def mesak_residuals(x,f,t,mmix,bc):
    p=x[0]
    q=x[1]
    w0=x[2]
    w1=x[3]
    w2=x[4]
    w3=x[5]
    m=x[6]
    
    #penalty enforces restrictions for w1
    #penalty = (w1*1e7)**2 if w1>0 else 0
    
    return np.sum((mesak_model(t,p,q,m,mmix,bc,w0,w1,w2,w3)-f)**2)+np.sum(np.abs(x))#+penalty

# #Marketing Mix Diffusion (MMD)
def agmm_model(t,p,q,m,mmix,bc,w0,w1,w2,w3,w4,w5,w6,eps=1e-2):  
    price=w1*mmix[:,1]+eps
    place=w2*mmix[:,2]+eps
    promotion=w3*np.sqrt(mmix[:,3])+eps
    sales=bc
    adoption=[]

    #Compute product trajectory approximation and sales using the previous product value
    product=[w6*w5*math.exp(-1/w4)]
    for i in t:
        if i == 1:
            radiation=p*price[i-1]*(m*place[i-1]-sales)*promotion[i-1]
            diffusion=q*product[-1]
            sales+=w0+radiation+diffusion
            adoption.append(max(0,sales))
            continue
        product.append(w6*math.exp(-i/w4)*solveVolterraJIT(lambda x,y: np.ones(y.shape),lambda x: np.divide(np.exp(x/w4)*f(x,product[-1],sales,p,q,m,w0,w1,w2,w3,w4,w5)*u_i(x),w4), 1, i)[-1,-1]+w5*math.exp(-i/w4))
        radiation=p*price[i-1]*(m*place[i-1]-sales)*promotion[i-1]
        diffusion=q*product[-1]
        sales+=w0+radiation+diffusion
        adoption.append(max(0,sales))
    return adoption

def agmm_residuals(x,f,t,mmix,bc):
    p=x[0]
    q=x[1]
    w0=x[2]
    w1=x[3]
    w2=x[4]
    w3=x[5]
    w6=x[6]
    m=x[7]
    loss= np.sum((agmm_model(t,p,q,m,mmix,bc,w0,w1,w2,w3,1e3,1e3,w6)-f)**2)+np.sum(np.abs(x)**2)
    #print(loss)
    #print("Additive Marketing Mix Diffusion training")
    #print("Loss: "+str(loss))
    return loss

bass_params=[]
mesak_params=[]
mm_params=[]
forecasts_monte_carlo=[]
enum=1

print("Start of Sensitivity Analysis.")
for adoption_iteration,mmix_iteration in zip(random_walks,rws_mmix):
    print("Iteration "+str(enum)+" out of 10,000.")
    adoption_iteration=np.array(adoption_iteration)
    mmix_iteration=np.array(mmix_iteration)
    
    bass_diffusion=least_squares(
        bass_residuals, 
        x0=[0.03,0.38,2e7], 
        args=(adoption_iteration[:60],t_train,bc),
        bounds=((0,0,1.5e7),(1,1,2e7))
        )
    
    p=bass_diffusion.x[0]
    q=bass_diffusion.x[1]
    m=bass_diffusion.x[2]
    bass_forecast_is=bass_model(t_train,p,q,m,bc)
    bass_forecast_oos=bass_model(t_test,p,q,m,adoption[60-1])
    bass_forecast=bass_forecast_is+bass_forecast_oos
    
    bass_params.append([p,q,m])

    mesak_diffusion=minimize(
        mesak_residuals, 
        method='nelder-mead', 
        x0=[1e-12,1e-12,-1e8,1,1,1,2e7], 
        bounds=(
            (0,1),
            (0,1),
            (-1,1),
            (-1,0),
            (0,1),
            (1.5e7,2e7)
            ),
        args=(adoption_iteration[:60],t_train,mmix_iteration,bc)
        )
    
    p=mesak_diffusion.x[0]
    q=mesak_diffusion.x[1]
    w0=mesak_diffusion.x[2]
    w1=mesak_diffusion.x[3]
    w2=mesak_diffusion.x[4]
    w3=mesak_diffusion.x[5]
    m=mesak_diffusion.x[6]
    mesak_forecast_is=mesak_model(t_train,p,q,m,mmix_iteration,bc,w0,w1,w2,w3)
    mesak_forecast_oos=mesak_model(t_test,p,q,m,mmix_iteration,adoption_iteration[60-1],w0,w1,w2,w3)
    mesak_forecast=mesak_forecast_is+mesak_forecast_oos

    mesak_params.append([p,q,w0,w1,w2,w3,m])    

    x0=[ 1.00000000e-002,  7.83110479e-101,  1.00000000e+000,
            -1.00000000e+000,  0.00000000e+000, -1.00000000e+000,
            3.96644485e-002, 2e7]
    
    agmm_diffusion=minimize(
        agmm_residuals, 
        method='nelder-mead',
        jac=lambda x,f,t,m,mmix_iteration,bc: approx_fprime(x, agmm_residuals, 1e-12,f,t,m,mmix_iteration,bc), 
        x0=x0,
        args=(adoption_iteration[:60],t_train,mmix_iteration,bc),
        options={'learning_rate':1e-12,'maxiter':100},
        bounds=(
            (0,1),
            (0,1),
            (-1,1),
            (-1,0),
            (0,1),
            (0,1),
            (-1,0),
            (1.5e7,2e7)
            )
        )
    
    p=agmm_diffusion.x[0]
    q=agmm_diffusion.x[1]
    w0=agmm_diffusion.x[2]
    w1=agmm_diffusion.x[3]
    w2=agmm_diffusion.x[4]
    w3=agmm_diffusion.x[5]
    w6=agmm_diffusion.x[6]
    m=agmm_diffusion.x[7]
    agmm_forecast_is=agmm_model(t_train,p,q,m,mmix_iteration,bc,w0,w1,w2,w3,1e3,1e3,w6)
    agmm_forecast_oos=agmm_model(t_test,p,q,m,mmix_iteration,adoption_iteration[60-1],w0,w1,w2,w3,1e3,1e3,w6)
    agmm_forecast=agmm_forecast_is+agmm_forecast_oos
    
    mm_params.append([p,q,w0,w1,w2,w3,w6,m])
    
    forecasts=pd.DataFrame({
        'Observed':adoption,
        'Bass Diffusion':bass_forecast,
        'Mesak Diffusion':mesak_forecast,
        'Marketing Mix Diffusion':agmm_forecast
        },index=t)
    
    forecasts_monte_carlo.append(forecasts)
    
    enum+=1
