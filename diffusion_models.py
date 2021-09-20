# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 21:53:23 2021

@author: MMSETUBAL
"""

from scipy.optimize import least_squares
from scipy.optimize import minimize, differential_evolution
from scipy.optimize import approx_fprime
from scipy.optimize import fmin
from scipy.optimize import minimize_scalar
import multiprocessing
import operator
import scipy.integrate as integrate
import pandas as pd
import numpy as np
import inteq
import math
import requests
import networkx as nx
from timeit import default_timer as timer
from numba import jit
import pdb
import copy
import random
import time

import warnings
warnings.filterwarnings("ignore")

#import sgd-for-scipy from github gist
exec(requests.get('https://gist.githubusercontent.com/lgpintomkt/9dbf22eb514d275cd89be1172477a1e8/raw/362a8c7f49d4c622af858202b6be0b3f7e33aca3/sgd-for-scipy.py').text)


def rad(x,t,m,mmix,bc): 
    p=x[0];q=x[1];w0=x[2];w1=x[3];w2=x[4];w3=x[5];
    return np.max(radiation_function(t,p,q,m,mmix,bc,w0,w1,w2,w3))

def simplify_graph(subgraph,percent=0.03):
    sbc=subgraph.copy()
    #remove edges with small weights until the graph becomes a DAG
    sorted_edges=sorted(sbc.edges(data=True), key=lambda t: t[2].get('weight', 1))
    noe=sbc.number_of_edges()
    non=sbc.number_of_nodes()
    print("Network size: "+str(noe)+" edges.")
    print("Removing edges...", end="")
    if non < 100:
        #too slow for large subgraphs
        for index,edge in enumerate(sorted_edges):
            node_i=edge[0]
            node_j=edge[1]
            sbc.remove_edge(node_i,node_j)
            #print("Removed "+str(index)+" edges.")
            if nx.is_directed_acyclic_graph(sbc):
                break
    else:
        #faster for large subgraphs
        edges=[(edge[0],edge[1]) for edge in sorted_edges]     
        n = int(len(edges) * percent) #3% of edges by default
        i=1
        while not(nx.is_directed_acyclic_graph(sbc)):
            #pdb.set_trace()
            to_remove=edges[:n*i]
            sbc.remove_edges_from(to_remove)
            print("...",end="")
            i+=1
    print("Removed "+str(len(to_remove))+" edges.")
    print("DAG Ok. Done.")
    return sbc

def diff(li1, li2):
    return list(set(li1) - set(li2))

def susceptibles(d1,ids,preds):
   d2 = []
   d3 = []
   for d in d1:
       d2 += preds[ids[d]]
       #d2 += G.successors(ids[d])
       d3.append(ids[d])
   return diff(list(set(d2)),d3)

def link_classes(G,L):
    degrees=G.out_degree(weight='weight')
    degs=[]
    for degree in degrees:
        degs.append(degree[1])
    degs.sort()
    degs=np.array_split(degs,L)
    link_classes=dict()
    for i,degree in enumerate(degs):
        link_classes[i]=min(degree)
    return link_classes

def centrality_heuristic(G,method='outdegree',k=100,output='id'):
    if method=='outdegree':
        centralities=dict(G.out_degree(weight='weight'))
    if method=='eigenvector':
        centralities=dict(nx.eigenvector_centrality(G,weight='weight'))
    if method=='closeness':
        centralities=dict(nx.closeness_centrality(G))
    names=nx.get_node_attributes(G,"name")
    influencers=[]
    
    for i in range(k):
        top_influencer=max(centralities.items(), key=operator.itemgetter(1))[0]
        if output=='name':
            influencers.append(names[top_influencer])
        elif output=='id':
            influencers.append(top_influencer)
        del centralities[str(top_influencer)]
    return influencers

def kempe_greedy_algorithm(G,threshold_model,t,*args,k=10):
    names=nx.get_node_attributes(G,"name")
    graph=copy.deepcopy(G)
    influencers=[]
    
    #modification to make it faster, consider only top 15
    #candidates=graph.nodes()
    #candidates=list(set(centrality_heuristic(graph,k=3,method='eigenvector')))
    candidates=list(set(centrality_heuristic(graph,k=15,method='eigenvector')+centrality_heuristic(graph,k=15,method='outdegree')))
    print("Considering "+str(len(candidates))+" candidates.")
    for i in range(1,k+1):
        activation=dict()    
        candidate=1
        #newgraph=copy.deepcopy(graph)
        
        nodes=G.nodes()
        for node in nodes:
            node_name=names[node]
            if node_name in influencers:
                #pdb.set_trace()
                continue
            if node not in candidates:
                node_name=names[node]
                activation[node]=0
                continue
            if i==1:
                #print("Computing baseline...")
                baseline=threshold_model(graph,[],t,influencers,*args[1:])[1][0]
            print("Influencer candidate "+str(candidate)+": "+node_name)
            #pdb.set_trace()
            new=threshold_model(graph,[node_name],t,influencers,*args[1:])[1][-1]
            activation[node]=new-baseline
            candidate+=1
            #print("k="+str(k)+" "+" n="+node+" ("+node_name+"): "+str(activation[node]))
        top_influencer=max(activation.items(), key=operator.itemgetter(1))[0]
        influencers.append(names[top_influencer])
        G.remove_node(top_influencer)
        baseline=max(activation.values())
        print("Found "+str(len(influencers))+" influencers out of "+str(k)+": "+names[top_influencer])
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
G=nx.read_graphml(data_path+"\\Data\\Network\\Network Files\\japan_municipalities_extended_normalized_v2.xml")
simplified=simplify_graph(G,0.01)

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

preds=dict()
for node in G.nodes():
    preds[node]=list(G.predecessors(node))

preds_simp=dict()
for node in simplified.nodes():
    preds_simp[node]=list(simplified.predecessors(node))

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

bass_diffusion=least_squares(
    bass_residuals, 
    x0=[0.03,0.38,2e7], 
    args=(adoption[:60],t_train,bc),
    bounds=((0,0,1.5e7),(1,1,2e7))
    )

p=bass_diffusion.x[0]
q=bass_diffusion.x[1]
m=bass_diffusion.x[2]
bass_forecast_is=bass_model(t_train,p,q,m,bc)
bass_forecast_oos=bass_model(t_test,p,q,m,adoption[60-1])
bass_forecast=bass_forecast_is+bass_forecast_oos

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
    args=(adoption[:60],t_train,mmix,bc)
    )

p=mesak_diffusion.x[0]
q=mesak_diffusion.x[1]
w0=mesak_diffusion.x[2]
w1=mesak_diffusion.x[3]
w2=mesak_diffusion.x[4]
w3=mesak_diffusion.x[5]
m=mesak_diffusion.x[6]
mesak_forecast_is=mesak_model(t_train,p,q,m,mmix,bc,w0,w1,w2,w3)
mesak_forecast_oos=mesak_model(t_test,p,q,m,mmix,adoption[60-1],w0,w1,w2,w3)
mesak_forecast=mesak_forecast_is+mesak_forecast_oos

# #General Marketing Mix Modelling (GMM)
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
        sales+=w0+(p*price[i-1]+q*product[-1])*(m*place[i-1]-sales)*promotion[i-1]
        adoption.append(max(0,sales))
        
    return adoption

def gmm_residuals(x,f,t,mmix,bc):
    p=x[0]
    q=x[1]
    w0=x[2]
    w1=x[3]
    w2=x[4]
    w3=x[5]
    w6=x[6]
    m=x[7]
    loss= np.sum((gmm_model(t,p,q,m,mmix,bc,w0,w1,w2,w3,1e3,1e3,w6)-f)**2)+np.sum(np.abs(x)**2)
    #print(loss)
    #print("Marketing Mix Diffusion training")
    print("Loss: "+str(loss))
    return loss

# gmm_diffusion=differential_evolution(
#     gmm_residuals, 
#     args=(adoption[:60],t_train,m,mmix,bc),
#     bounds=(
#         (0,1e-2),
#         (0,1e-100),
#         (0,1),
#         (-1,0),
#         (0,1),
#         (0,1),
#         (1,500),
#         (1,500),
#         (0,1e-100),
#         )
#     )

x0=[ 1.00000000e-002,  7.83110479e-101,  1.00000000e+000,
        -1.00000000e+000,  0.00000000e+000, -1.00000000e+000,
        3.96644485e-002,  3.15104987e-002,  -6.05183820e-101,2e7]

gmm_diffusion=minimize(
    gmm_residuals, 
    method='nelder-mead',
    jac=lambda x,f,t,m,mmix,bc: approx_fprime(x, agmm_residuals, 1e-12,f,t,m,mmix,bc), 
    x0=x0,
    args=(adoption[:60],t_train,mmix,bc),
    options={'learning_rate':1e-12,'maxiter':100},
    #options={'learning_rate':1e-12,'maxiter':500},
        bounds=(
        (0,1),
        (0,1),
        (-1,1),
        (-1,0),
        (0,1),
        (0,1),
        (-1,0)
        )
    )


# x0=gmm_diffusion.x

p=gmm_diffusion.x[0]
q=gmm_diffusion.x[1]
w0=gmm_diffusion.x[2]
w1=gmm_diffusion.x[3]
w2=gmm_diffusion.x[4]
w3=gmm_diffusion.x[5]
w6=gmm_diffusion.x[6]
m=gmm_diffusion.x[7]
gmm_forecast_is=gmm_model(t_train,p,q,m,mmix,bc,w0,w1,w2,w3,1e3,1e3,w6)
gmm_forecast_oos=gmm_model(t_test,p,q,m,mmix,adoption[60-1],w0,w1,w2,w3,1e3,1e3,w6)
gmm_forecast=gmm_forecast_is+gmm_forecast_oos

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
    print("Loss: "+str(loss))
    return loss

#Use genetic/evolutionary algorithm to find initial values for parameters
# agmm_diffusion=differential_evolution(
#     agmm_residuals, 
#     args=(adoption[:60],t_train,m,mmix,bc),
#     bounds=(
#         (0,1e-2),
#         (0,1e-100),
#         (0,1),
#         (0,1),
#         (0,1),
#         (-1,0),
#         (1,500),
#         (1,500),
#         (0,1e-100),
#         )
#     )

x0=[ 1.00000000e-002,  7.83110479e-101,  1.00000000e+000,
        -1.00000000e+000,  0.00000000e+000, -1.00000000e+000,
        3.96644485e-002,  3.15104987e-002,  -6.05183820e-101, 2e7]

agmm_diffusion=minimize(
    agmm_residuals, 
    method='nelder-mead',
    jac=lambda x,f,t,m,mmix,bc: approx_fprime(x, agmm_residuals, 1e-12,f,t,m,mmix,bc), 
    x0=x0,
    args=(adoption[:60],t_train,mmix,bc),
    options={'learning_rate':1e-12,'maxiter':100},
    #options={'learning_rate':1e-12,'maxiter':500},
    bounds=(
        (0,1),
        (0,1),
        (-1,1),
        (-1,0),
        (0,1),
        (0,1),
        (-1,0)
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
agmm_forecast_is=agmm_model(t_train,p,q,m,mmix,bc,w0,w1,w2,w3,1e3,1e3,w6)
agmm_forecast_oos=agmm_model(t_test,p,q,m,mmix,adoption[60-1],w0,w1,w2,w3,1e3,1e3,w6)
agmm_forecast=agmm_forecast_is+agmm_forecast_oos


forecasts=pd.DataFrame({
    'Observed':adoption,
    'Bass Diffusion':bass_forecast,
    'Mesak Diffusion':mesak_forecast,
    'General Marketing Mix Model':gmm_forecast,
    'Additive General Marketing Mix Model':agmm_forecast
    },index=t)

forecasts.to_excel(data_path+"\\results.xlsx")

#Optimal Control
import control
from control import optimal
from scipy.optimize import LinearConstraint

budget=mmix[60:72,:].sum()

X0=-adoption[60-1]

constraints=[
    (LinearConstraint(np.identity(4),0,1),np.identity(4),0,1),
    (LinearConstraint(np.ones(4),0,budget),np.ones(4),0,budget)
    ]

def cost(t,x,u):
    return u.sum() #cost of the marketing mix variables

def gmm_dynamics(t, x, u, params, eps=1e-2, w4=1e3, w5=1e3):
    p=params.get('Coefficient of Innovation')
    q=params.get('Coefficient of Innovation')
    m=params.get('Market Potential')
    w0=params.get('Baseline Adoption')
    w1=params.get('Price Elasticity')
    w2=params.get('Distribution Intensity')
    w3=params.get('Advertising Investment')
    w6=params.get('Product Quality Impact')
    price=w1*u[1]+eps
    place=w2*u[2]+eps
    promotion=w3*np.sqrt(u[3])+eps
    sales=x

    #Compute product trajectory approximation and sales using the previous product value
    product=[w6*w5*math.exp(-1/w4)]
    
    if t>1:
        for i in range(2,t):
            product.append(w6*math.exp(-i/w4)*solveVolterraJIT(lambda x,y: np.ones(y.shape),lambda x: np.divide(np.exp(x/w4)*f(x,product[-1],sales,p,q,m,w0,w1,w2,w3,w4,w5)*u_i(x),w4), 1, i)[-1,-1]+w5*math.exp(-i/w4))
            
    product=product[-1]
    
    sales+=w0+(p*price+q*product)*(m*place-sales)*promotion
    adoption=sales
    return adoption

def agmm_dynamics(t, x, u, params, eps=1e-2, w4=1e3, w5=1e3):
    p=params.get('Coefficient of Innovation')
    q=params.get('Coefficient of Innovation')
    m=params.get('Market Potential')
    w0=params.get('Baseline Adoption')
    w1=params.get('Price Elasticity')
    w2=params.get('Distribution Intensity')
    w3=params.get('Advertising Investment')
    w6=params.get('Product Quality Impact')
    price=w1*u[1]+eps
    place=w2*u[2]+eps
    promotion=w3*np.sqrt(u[3])+eps
    sales=x

    #Compute product trajectory approximation and sales using the previous product value
    product=[w6*w5*math.exp(-1/w4)]
    
    if t>1:
        for i in range(2,t):
            product.append(w6*math.exp(-i/w4)*solveVolterraJIT(lambda x,y: np.ones(y.shape),lambda x: np.divide(np.exp(x/w4)*f(x,product[-1],sales,p,q,m,w0,w1,w2,w3,w4,w5)*u_i(x),w4), 1, i)[-1,-1]+w5*math.exp(-i/w4))
        
    product=product[-1]
    
    if t == 1:
        radiation=p*price*(m*place-sales)*promotion
        diffusion=q*product
        sales+=w0+radiation+diffusion
        adoption=sales
        return adoption
    radiation=p*price*(m*place-sales)*promotion
    diffusion=q*product
    sales+=w0+radiation+diffusion
    adoption=sales
    return adoption

#Using MMD
p=gmm_diffusion.x[0]
q=gmm_diffusion.x[1]
w0=gmm_diffusion.x[2]
w1=gmm_diffusion.x[3]
w2=gmm_diffusion.x[4]
w3=gmm_diffusion.x[5]
w6=gmm_diffusion.x[6]
m=gmm_diffusion.x[7]

X0=adoption[60-1]
sys=control.NonlinearIOSystem(updfcn=gmm_dynamics,
                              inputs=['Product','Price','Place','Promotion'], 
                              outputs=['Adoption'], 
                              dt=1, 
                              states=['Adoption'],
                              name="Dynamic Marketing System",  
                              params={
                                  'Coefficient of Innovation':p, 
                                  'Coefficient of Imitation':q, 
                                  'Market Potential': m,
                                  'Baseline Adoption': w0,
                                  'Price Elasticity': w1,
                                  'Distribution Intensity': w2,
                                  'Advertising Investment': w3,
                                  'Product Quality Impact': w6
                                  })
res_gmm=optimal.solve_ocp(sys, t_test, X0, cost, constraints)
