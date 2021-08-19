# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 21:53:23 2021

@author: MMSETUBAL
"""

from scipy.optimize import least_squares
from scipy.optimize import minimize, differential_evolution
from scipy.optimize import approx_fprime
from scipy.optimize import fmin
import scipy.integrate as integrate
import pandas as pd
import numpy as np
import inteq
import math
import requests
from timeit import default_timer as timer
from numba import jit
import pdb

#import sgd-for-scipy from github gist
exec(requests.get('https://gist.githubusercontent.com/lgpintomkt/9dbf22eb514d275cd89be1172477a1e8/raw/2898c5c73db13936baa7bb673a73f485bb151a66/sgd-for-scipy.py').text)

def link_classes(G):
    degrees=G.out_degree(weight='weight')
    degs=[]
    for degree in degrees:
        degs.append(degree[1])
    degs.sort()
    degs=np.array_split(degs,L)
    link_classes=dict()
    for i,degree in enumerate(degs):
        link_classes[i]=max(degree)
    return link_classes

def kempe_greedy_algorithm(G,threshold_model,k=10,steps=10):
    names=nx.get_node_attributes(G,"name")
    graph=copy.deepcopy(G)
    influencers=[]
    for i in range(k):
        activation=dict()
        for node in graph.nodes():
            node_name=names[node]
            activation[node]=sum(threshold_model(graph,influencers+[node_name],steps)[1].values())-sum(continuous_threshold_model(graph,influencers,steps)[1].values())
            #print("k="+str(k)+" "+" n="+node+" ("+node_name+"): "+str(activation[node]))
        top_influencer=max(activation.items(), key=operator.itemgetter(1))[0]
        influencers.append(names[top_influencer])
        graph.remove_node(top_influencer)
    return influencers

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


solveVolterraJIT=jit(inteq.SolveVolterra)
integrate_quadJIT=jit(integrate.quad)

data_path="C:\\Users\\MMSETUBAL\\Desktop\\Artigos\\MMM and Information Diffusion over Complex Networks"
data=pd.read_csv(data_path+"\\Data\\Statistical\\mmm_data.csv",sep=";").set_index("t")
G=nx.read_graphml(data_path+"\\Data\\Network\\Network Files\\japan_municipalities_extended.xml")

#Market Potential - Mobile Broadband Subscriptions (Japan Dec. 2019)
m=data['Adoption'].iloc[-1]
#Boundary condition: cumulative sales on t=1
bc=11197800 #cumulative subscriptions in Dec2013 - see https://data.oecd.org/broadband/mobile-broadband-subscriptions.htm
adoption=np.array(data['Adoption'])
t=np.array(data.index)
t_train=np.array(data.index)[:60]
t_test=np.array(data.index)[60:]
mmix=np.array(data[['Product','Price','Place','Promotion']])

#Bass Diffusion (BD)
def bass_model(t,p,q,m,bc):
    sales=bc
    adoption=[]
    for i in t:
        sales+=p*m+(q-p)*sales-(q/m)*(sales**2)
        adoption.append(max(0,sales))
    return adoption

def bass_residuals(x,f,t,m,bc):
    p=x[0]
    q=x[1]
    return bass_model(t,p,q,m,bc)-f

bass_diffusion=least_squares(
    bass_residuals, 
    x0=[0.03,0.38], 
    args=(adoption,t,m,bc)
    )

p=bass_diffusion.x[0]
q=bass_diffusion.x[1]
bass_forecast=bass_model(t,p,q,m,bc)

#Mesak Diffusion (MD)
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

def mesak_residuals(x,f,t,m,mmix,bc):
    p=x[0]
    q=x[1]
    w0=x[2]
    w1=x[3]
    w2=x[4]
    w3=x[5]
    return np.sum((mesak_model(t,p,q,m,mmix,bc,w0,w1,w2,w3)-f)**2)+np.sum(np.abs(x))

mesak_diffusion=minimize(
    mesak_residuals, 
    method='nelder-mead', 
    x0=[1e-12,1e-12,1e8,-1,1,1], 
    args=(adoption,t,m,mmix,bc)
    )

p=mesak_diffusion.x[0]
q=mesak_diffusion.x[1]
w0=mesak_diffusion.x[2]
w1=mesak_diffusion.x[3]
w2=mesak_diffusion.x[4]
w3=mesak_diffusion.x[5]
mesak_forecast=mesak_model(t,p,q,m,mmix,bc,w0,w1,w2,w3)

#General Marketing Mix Modelling (GMM)
def gmm_model(t,p,q,m,mmix,bc,w0,w1,w2,w3,w4,w5,w6,eps=1e-2):  
    price=w1*mmix[:,1]+eps
    place=w2*mmix[:,2]+eps
    promotion=w3*np.sqrt(mmix[:,3])+eps
    sales=bc
    adoption=[]

    #Compute product trajectory approximation and sales using the previous product value
    product=[w6*w5*math.exp(-1/w4)]
    for i in t:
        if i == 1:
            sales+=w0+(p*price[i-1]+q*product[i-1])*(m*place[i-1]-sales)*promotion[i-1]
            adoption.append(max(0,sales))
            continue
        product.append(w6*math.exp(-i/w4)*solveVolterraJIT(lambda x,y: np.ones(y.shape),lambda x: np.divide(np.exp(x/w4)*f(x,product[-1],sales,p,q,m,w0,w1,w2,w3,w4,w5)*u_i(x),w4), 1, i)[-1,-1]+w5*math.exp(-i/w4))
        sales+=w0+(p*price[i-1]+q*product[i-1])*(m*place[i-1]-sales)*promotion[i-1]
        adoption.append(max(0,sales))
        
    return adoption

def gmm_residuals(x,f,t,m,mmix,bc):
    p=x[0]
    q=x[1]
    w0=x[2]
    w1=x[3]
    w2=x[4]
    w3=x[5]
    w6=x[6]
    loss= np.sum((gmm_model(t,p,q,m,mmix,bc,w0,w1,w2,w3,1e3,1e3,w6)-f)**2)+np.sum(np.abs(x)**2)
    #print(loss)
    return loss

gmm_diffusion=differential_evolution(
    gmm_residuals, 
    args=(adoption,t,m,mmix,bc),
    bounds=(
        (0,1e-2),
        (0,1e-100),
        (0,1),
        (0,1),
        (0,1),
        (-1,0),
        (1,500),
        (1,500),
        (0,1e-100),
        )
    )

x0=gmm_diffusion.x

gmm_diffusion=minimize(
    gmm_residuals, 
    method=adam,
    #method='nelder-mead',
    jac=lambda x,f,t,m,mmix,bc: approx_fprime(x, gmm_residuals, 1e-12,f,t,m,mmix,bc), 
    x0=[-9.99491090e-04, -8.45522527e-04,  1.00000000e-12,  8.94036601e-04,
        -6.72656593e-04,  1.85468863e-04, -1.84005389e-04],
    #x0=x0,
    args=(adoption,t,m,mmix,bc),
    options={'learning_rate':1e-6,'maxiter':500},
        bounds=(
        (0,1e-2),
        (0,1e-100),
        (0,1),
        (0,1),
        (0,1),
        (-1,0),
        (1,500),
        (1,500),
        (0,1e-100),
        )
    )


p=gmm_diffusion.x[0]
q=gmm_diffusion.x[1]
w0=gmm_diffusion.x[2]
w1=gmm_diffusion.x[3]
w2=gmm_diffusion.x[4]
w3=gmm_diffusion.x[5]
w6=gmm_diffusion.x[6]
gmm_forecast=gmm_model(t,p,q,m,mmix,bc,w0,w1,w2,w3,1e3,1e3,w6)

#Additive GMM (AGMM)
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

def agmm_residuals(x,f,t,m,mmix,bc):
    p=x[0]
    q=x[1]
    w0=x[2]
    w1=x[3]
    w2=x[4]
    w3=x[5]
    w6=x[6]
    loss= np.sum((agmm_model(t,p,q,m,mmix,bc,w0,w1,w2,w3,1e3,1e3,w6)-f)**2)+np.sum(np.abs(x)**2)
    #print(loss)
    return loss

#Use genetic/evolutionary algorithm to find initial values for parameters
agmm_diffusion=differential_evolution(
    agmm_residuals, 
    args=(adoption,t,m,mmix,bc),
    bounds=(
        (0,1e-2),
        (0,1e-100),
        (0,1),
        (0,1),
        (0,1),
        (-1,0),
        (1,500),
        (1,500),
        (0,1e-100),
        )
    )

x0=agmm_diffusion.x

agmm_diffusion=minimize(
    agmm_residuals, 
    method='nelder-mead',
    jac=lambda x,f,t,m,mmix,bc: approx_fprime(x, agmm_residuals, 1e-12,f,t,m,mmix,bc), 
    x0=x0,
    args=(adoption,t,m,mmix,bc),
    #options={'learning_rate':1e-12,'maxiter':100}
    options={'learning_rate':1e-12,'maxiter':500},
        bounds=(
        (0,1e-2),
        (0,1e-100),
        (0,1),
        (0,1),
        (0,1),
        (-1,0),
        (1,500),
        (1,500),
        (0,1e-100),
        )
    )

p=agmm_diffusion.x[0]
q=agmm_diffusion.x[1]
w0=agmm_diffusion.x[2]
w1=agmm_diffusion.x[3]
w2=agmm_diffusion.x[4]
w3=agmm_diffusion.x[5]
w6=agmm_diffusion.x[6]
agmm_forecast=agmm_model(t,p,q,m,mmix,bc,w0,w1,w2,w3,1e3,1e3,w6)

#Additive GMM over Complex Networks (AGMM-CN)
#Hyperparameters
K=8     #Influencer seed set size K
L=10    #Number of link classes to consider

def radiation_function(t,p,m,mmix,bc,w0,w1,w2,w3,eps=1e-2):
    price=w1*mmix[:,1]+eps
    place=w2*mmix[:,2]+eps
    promotion=w3*np.sqrt(mmix[:,3])+eps
    
    return w0+p*price[t-1]*(m*place[t-1]-bc)*promotion[t-1]

def diffusion_function(sales,t,p,m,mmix,bc,w0,w1,w2,w3,w4,w5,w6,lw,eps=1e-2):  
    adoption=[]
    
    #Compute product trajectory approximation and sales using the previous product value
    product=[w6*w5*math.exp(-1/w4)]
    for i in t:
        if i == 1:
            diffusion=product[-1]
            adoption.append(diffusion)
            continue
        product.append(w6*math.exp(-i/w4)*solveVolterraJIT(lambda x,y: np.ones(y.shape),lambda x: np.divide(np.exp(x/w4)*f(x,product[-1],sales,p,q,m,w0,w1,w2,w3,w4,w5)*u_i(x),w4), 1, i)[-1,-1]+w5*math.exp(-i/w4))
        diffusion=product[-1]
    return diffusion

def marketing_threshold_model(G,seeds,t,p,m,mmix,w0,w1,w2,w3,w6,thres,mode='general'):
    steps=len(t)
    actives=copy.deepcopy(seeds)
    name_dict=nx.get_node_attributes(G,"name")
    link_classes=nx.get_node_attributes(G,"link_class")
    thresholds=dict()
    for node in G.nodes():
        for index,link in enumerate(thres):
            if link_classes[node] == link:
                if mode=='general':
                    thresholds[node]=thres[index]
                elif mode == 'submodular':
                    thresholds[node]=thres[index]-maximum_radiation
        
    influence_values=dict()
    for node in G.nodes:
        influence_values[node]=1e-20
    for step in range(steps):
        radiation_level=radiation_function(t,p,q,m,mmix,w0,w1,w2,w3)
        for node_i in G.nodes:
            if name_dict[node_i] in actives:
                continue
            influence_value=1e-20
            for node_j in G.predecessors(node_i):
                if name_dict[node_j] in actives:
                    influence_value+=G.get_edge_data(node_j,node_i)['weight']*influence_values[node_j]
            if mode == 'general':
                influence_values[node_i]=radiation_level+diffusion_function(influence_value,step)
            elif mode == 'submodular':
                influence_values[node_i]=diffusion_function(influence_value,step)
        for node_i in G.nodes:
            if influence_values[node_i]>thresholds[node_i]:
                actives.append(name_dict[node_i])
                
    return sum(influence_values.values())

def submodular_marketing_threshold_model(G,seeds,t,p,m,mmix,w0,w1,w2,w3,w6,thres,maximum_radiation):
    marketing_threshold_model(G,seeds,t,p,m,mmix,w0,w1,w2,w3,w6,thres,maximum_radiation,mode='submodular')

#initialize link class parameterization
link_class=link_classes(G)
degrees=G.out_degree(weight='weight')
assigned_classes=dict()
for node in G.nodes():
    for link in list(link_class.keys()):
        if degrees[node]==0:
            assigned_classes[node]=str(0)
        if degrees[node]>=link_class[link]:
            assigned_classes[node]=str(link)
nx.set_node_attributes(G,assigned_classes,'link_class')


#find K top influencers using submodular marketing threshold model
influencers=kempe_greedy_algorithm(G,marketing_threshold_model,k=K)



def agmm_cn_residuals(x,f,t,m,mmix,bc,G,lc,seeds):
    p=x[0]
    w0=x[2]
    w1=x[3]
    w2=x[4]
    w3=x[5]
    w6=x[6]
    loss= np.sum((general_marketing_mix_threshold_model(G,seeds,t,p,m,mmix,w0,w1,w2,w3,1e3,1e3,w6,x[6:])-f)**2)+np.sum(np.abs(x)**2)
    print(loss)
    return loss

#Use genetic/evolutionary algorithm to find initial values for parameters
agmm_cn_diffusion=differential_evolution(
    agmm_residuals, 
    args=(adoption,t,m,mmix,bc,G,list(link_class.keys())),
    bounds=(
        (0,1e-2),
        (0,1e-100),
        (0,1),
        (0,1),
        (0,1),
        (-1,0),
        (1,500),
        (1,500),
        (0,1e-100),
        )
    )

x0=agmm_diffusion.x

agmm_cn_diffusion=minimize(
    agmm_cn_residuals, 
    method='nelder-mead',
    jac=lambda x,f,t,m,mmix,bc: approx_fprime(x, agmm_residuals, 1e-12,f,t,m,mmix,bc), 
    x0=x0,
    args=(adoption,t,m,mmix,bc),
    options={'learning_rate':1e-12,'maxiter':500},
    bounds=(
    (0,1e-2),
    (0,1e-100),
    (0,1),
    (0,1),
    (0,1),
    (-1,0),
    (1,500),
    (1,500),
    (0,1e-100),
    )
    )
    
p=agmm_cn_diffusion.x[0]
q=agmm_cn_diffusion.x[1]
w0=agmm_cn_diffusion.x[2]
w1=agmm_cn_diffusion.x[3]
w2=agmm_cn_diffusion.x[4]
w3=agmm_cn_diffusion.x[5]
w6=agmm_cn_diffusion.x[6]
agmm_forecast=agmm_model(t,p,q,m,mmix,bc,w0,w1,w2,w3,1e3,1e3,w6)


forecasts=pd.DataFrame({
    'Observed':adoption,
    'Bass Diffusion':bass_forecast,
    'Mesak Diffusion':mesak_forecast,
    'General Marketing Mix Model':gmm_forecast,
    'Additive General Marketing Mix Model':agmm_forecast
    },index=t)

forecasts.to_excel(data_path+"\\results.xlsx")
