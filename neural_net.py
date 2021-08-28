# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 09:51:01 2021

@author: MMSETUBAL
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import networkx as nx
import random
import operator
import scipy
import pandas as pd
from numba import jit
from scipy.optimize import minimize, differential_evolution

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

data_path="C:\\Users\\MMSETUBAL\\Desktop\\Artigos\\MMM and Information Diffusion over Complex Networks"
G=nx.read_graphml(data_path+"\\Data\\Network\\Network Files\\japan_municipalities_extended_normalized_v2.xml")

L=10

link_class=link_classes(G,L)
degrees=G.out_degree(weight='weight')
link_class_thres=dict()
link_class_thres[0]=random.uniform(1e10,1.5e10)
for link in list(link_class.keys())[1:]:
    link_class_thres[link]=random.uniform(0,1e12)/random.uniform(1,1e6)
    while link_class_thres[link]>link_class_thres[link-1]:
        link_class_thres[link]=random.uniform(0,1e15)/random.uniform(1,1e6)
assigned_classes=dict()
for node in G.nodes():
    for link in list(link_class.keys()):
        if degrees[node]==0:
            assigned_classes[node]=str(0)
        if degrees[node]>=link_class[link]:
            assigned_classes[node]=str(link)
nx.set_node_attributes(G,assigned_classes,'link_class')
thresholds=list(link_class_thres.values())

link_classes=nx.get_node_attributes(G,"link_class")


p=tf.Variable(5e-3,trainable=True,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
q=tf.Variable(5e-3,trainable=True,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
w0=tf.Variable(5e-3,trainable=True,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
w1=tf.Variable(5e-3,trainable=True,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
w2=tf.Variable(5e-3,trainable=True,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
w3=tf.Variable(5e-3,trainable=True,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
w4=tf.Variable(1,trainable=True,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
w5=tf.Variable(1,trainable=True,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
w6=tf.Variable(0.5,trainable=True,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
lamb=tf.Variable(2,trainable=True,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)



bc=11197800

influencers=centrality_heuristic(G,k=5,method='eigenvector',output="id")+centrality_heuristic(G,k=5,method='outdegree',output="id")
influencers=[int(i) for i in influencers]

initial_values=np.zeros([1889,1889])
for influencer in influencers:
    initial_values[influencer,:]=bc/len(influencers)

A=nx.to_numpy_array(G,weight='normalized weight')  
network=tf.keras.layers.Dense(
    1889, use_bias=False,
    kernel_initializer=tf.compat.v1.keras.initializers.Constant(A),
    trainable=False,dtype=tf.float32
)


data=pd.read_csv(data_path+"\\Data\\Statistical\\mmm_data.csv",sep=";").set_index("t")
m=data['Adoption'].iloc[-1]
adoption=np.array(data['Adoption'])
t=np.array(data.index)
t_train=np.array(data.index)[:55]
t_val=np.array(data.index)[55:60]
t_test=np.array(data.index)[60:]
mmix=np.array(data[['Product','Price','Place','Promotion']])

def marketing_mix_cn_model(t,G,bc,m,mmix,thres,influencers):

    global network    
    global link_classes
    thresholds=dict()
    for node in G.nodes():
        for index,link in enumerate(thres):
            if link_classes[node] == str(index):
                thresholds[node]=thres[index]
    thres_var=tf.Variable(np.tile(np.array(list(thresholds.values()))*-1,[1889,1]),trainable=False,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
    
    values=tf.clip_by_value(tf.nn.relu(network(influencers)+thres_var),0,1e2)
    u_0=w5*np.exp(-1/w4)
    f_i=np.zeros([1889,1889,73])
    f_i[:,:,0]=tf.multiply(np.exp(1/w4),values)+w5*np.exp(-1/w4)
    u=np.zeros([1889,73])
    u[:,0]=u_0
    adopt=np.zeros([1889,72])
    pred=np.zeros(len(t))
    activation=np.zeros(73)
    activation[0]=tf.reduce_sum(tf.cast(tf.math.greater(values[:,i-1], 0.), tf.float32))
    for i in t:
        radiation=w0+p*(-w1)*((mmix[i,1]+1e-12)**(-lamb))*(m*w2*mmix[i,2]-bc)*w3*np.sqrt(mmix[i,3])
        f_i[:,:,i]=tf.multiply(np.exp(i/w4),values)+w5*np.exp(-i/w4)
        if i==1:
            u[:,i]=w5*np.exp(-t[0]/w4)
        else:
            u[:,i]=w6*np.exp(-t[i-1]/w4)*tf.cast(tfp.math.trapz(np.exp(-t[0:i-1]/w4)*np.divide(np.exp(t[0:i-1]/w4)*tf.reduce_sum(tf.nn.relu(f_i[:,:,0:i-1]))*mmix[0:i-1,0],w4)),dtype=tf.float32)+w5*np.exp(-t[i-1]/w4)
        diffusion=q*u[:,i]
        adopt[:,i-1]=radiation+diffusion
        pred[i-1]=tf.reduce_sum(adopt[:,i-1])
        values=tf.clip_by_value(tf.nn.relu(tf.transpose(network(values))),0,1e2)
        activation[i]=tf.reduce_sum(tf.cast(tf.math.greater(values[:,i-1], 0.), tf.float32))
        tf.print("Step: ", i, ", Adoption: ", pred[i-1], " Activation: ", tf.math.reduce_max(activation))
        
            
    return pred
    

def marketing_mix_cn_residuals(x):
    p.assign(x[0])
    q.assign(x[1])
    w0.assign(x[2])
    w1.assign(x[3])
    w2.assign(x[4])
    w3.assign(x[5])
    w4.assign(x[6])
    w5.assign(x[7])
    w6.assign(x[8])
    lamb.assign(x[9])
    thresholds=[]
    for i in range(10):
        thresholds.append(x[9+i])
    
    global t_train
    global G
    global bc
    global m
    global mmix
    global adoption
    global initial_values
    loss=tf.reduce_sum((adoption[0:55] - marketing_mix_cn_model(t_train,G,bc,m,mmix,thresholds,initial_values))**2)
    tf.print("Loss: ",loss)
    print("")
    return loss

marketing_mix_cn_diffusion=minimize(
    marketing_mix_cn_residuals, 
    method='nelder-mead', 
    x0=[5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,1,1,0.5,2]+thresholds, 
    )
