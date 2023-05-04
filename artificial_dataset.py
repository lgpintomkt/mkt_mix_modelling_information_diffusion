# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 10:47:58 2022

@author: MMSETUBAL
"""

from pyDOE import *
import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
gpu=tf.config.list_physical_devices('GPU')[0]
tf.config.set_logical_device_configuration(gpu,[tf.config.LogicalDeviceConfiguration(memory_limit=17000)])

def marketing_mix_cn_model(x,steps,G,bc,mmix,thres,net,prev_pred=None,history=None,test=False):
    global network    
    global link_classes
    global pop_norm
    
    p.assign(x[0])
    q.assign(x[1])
    w0.assign(x[2])
    w1.assign(x[3])
    w2.assign(x[4])
    w3.assign(x[5])
    w4.assign(x[6])
    w5.assign(x[7])
    w6.assign(x[8])
    m.assign(x[9])

    if steps[0]==1:
        t_0=1
        are_influencers=list(bc>0)
        thresholds=dict()
        for node,is_influencer in zip(G.nodes(),are_influencers):
            for index,link in enumerate(thres):
                if link_classes[node] == str(index):
                    if is_influencer:
                        thresholds[node]=0
                    else:
                        thresholds[node]=thres[index]
        thres_var=tf.Variable(np.array(list(thresholds.values()))*-1,trainable=True,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
        state=bc+thres_var
    else:
        t_0=steps[0]
        steps=np.array(range(1,steps[-1]+1))
        thresholds=dict()
        for node in G.nodes():
            for index,link in enumerate(thres):
                if link_classes[node] == str(index):
                    thresholds[node]=thres[index]
        state=bc
    if history is None:
        f_t=np.zeros([G.number_of_nodes(),73])
        f_t[:,0]=tf.nn.relu(state)
    else:
        f_t=history
    
    pred=np.zeros(len(steps)+1)

    if prev_pred is None:
        prev_pred=tf.cast(bc.sum(),tf.float32)

    potential=m*tf.cast(pop_norm,tf.float32)
    activation=np.zeros(73)
    
    
    for t in steps[t_0-1:]:
        R=w0+p*w1*(mmix[t-1,1]+1e-12)*(m*w2*mmix[t-1,2]-prev_pred)*w3*(mmix[t-1,3]+1e-12)
        U=tf.cast(w5*np.exp(-t/w4),tf.float32)+tf.cast(tf.reshape(np.exp(-t/w4)*tfp.math.trapz(np.exp(-steps[0:t-1]/w4)*np.divide(np.exp(steps[0:t-1]/w4)*f_t[:,0:t-1]*mmix[0:t-1,0],w4)),[1,G.number_of_nodes()]),tf.float32)
        S=tf.clip_by_value(net(q*tf.nn.relu(tf.cast(tf.cast(U,tf.float32),tf.float32))),0,potential)
        f_t[:,t]=tf.nn.relu(state+potential*tf.nn.tanh(S+R))
        
        delta=f_t[:,t]-f_t[:,t-1]

        state+=delta
        actives=tf.nn.relu(state)
        pred[t]=tf.reduce_sum(actives)
        prev_pred=pred[t]
        activation[t]=tf.reduce_sum(tf.cast(tf.math.greater(actives, 0.), tf.float32))

        if test:
            tf.print("Step: ", t, " Adoption:", pred[t], " Activation:", activation[t])
    return pred[t_0:],state,f_t


#Variables
alpha=tf.Variable(5e-3,trainable=False,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
beta=tf.Variable(5e-3,trainable=False,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
p=tf.Variable(5e-3,trainable=False,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
q=tf.Variable(5e-5,trainable=False,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
w0=tf.Variable(0,trainable=False,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
w1=tf.Variable(1,trainable=False,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
w2=tf.Variable(1,trainable=False,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
w3=tf.Variable(1,trainable=False,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
w4=tf.Variable(1,trainable=False,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
w5=tf.Variable(1,trainable=False,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
w6=tf.Variable(1,trainable=False,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)

#MMIX Data
data_path="C:\\Users\\MMSETUBAL\\Desktop\\Artigos\\MMM and Information Diffusion over Complex Networks"
data=pd.read_csv(data_path+"\\Data\\Statistical\\mmm_data.csv",sep=";").set_index("t")
mmix=np.array(data[['Product','Price','Place','Promotion']])

#Constants
#m = number of nodes / market constant / potential market
m=tf.Variable(5000,trainable=False,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
#M = number of threshold classes
M=100
#N = number of influencer nodes / seed nodes = m/M
N=m/M
#K nearest neighbor connections
K=15
#Starting threshold
Z=25
#Boundary condition
#Assuming nodes represent individuals
bc=1
#Time steps
T=50
t=np.array(range(1,T+1))
#Pop norm global var
pop_norm=tf.Variable(tf.ones(int(m.numpy())))/m.numpy()

#Generate threshold classes
link_class_thres=dict()
link_class_thres[0]=Z
for link in range(1,M-1):
    link_class_thres[link]=max(1,link_class_thres[link-1]-3)
#Influencer class
link_class_thres[link+1]=0
thresholds=list(link_class_thres.values())

#Latin Hypercube Design
samples=lhs(8, samples=5, criterion='maximin')

nets_ws=[]
nets_sf=[]
sims_ws=[]
sims_sf=[]
for sample in samples:
    alpha.assign(sample[0])
    p.assign(sample[1])
    q.assign(sample[2]*1e-2)
    w0.assign(sample[3])
    w1.assign(sample[4]*1e-2)
    w2.assign(sample[5])
    w3.assign(sample[6])
    w6.assign(sample[7])

    #Generate Watts Strogatz Network
    ws=nx.watts_strogatz_graph(int(m.numpy()), K, alpha.numpy())
    sf=nx.scale_free_graph(int(m.numpy()), alpha.numpy(), 1-(alpha.numpy()+0.01), 0.01)
    
    #Assign Thresholds randomly    
    nodes=list(ws.nodes())
    link_classes=dict()
    initial_values=np.zeros([int(m.numpy())])
    for node in nodes:
        link_class=random.choice(list(range(M)))
        link_classes[node]=str(link_class)
        if link_class==0:
            initial_values[node]=bc
    nx.set_node_attributes(ws,link_classes,'link_class')
    
    #Set neural network graph layer from Watts Strogatz Network Adjacency
    A_ws=nx.to_numpy_array(ws)  

    network_ws=tf.keras.layers.Dense(
        int(m.numpy()), use_bias=False,
        kernel_initializer=tf.compat.v1.keras.initializers.Constant(A_ws),
        trainable=False,dtype=tf.float32
    )
    
    x=[p.numpy(),q.numpy(),w0.numpy(),w1.numpy(),w2.numpy(),w3.numpy(),w4.numpy(),w5.numpy(),w6.numpy(),m.numpy()]+thresholds
    #Run Simulation
    sim,_,_=marketing_mix_cn_model(x,t,ws,initial_values,mmix,thresholds,network_ws,test=False)
    sim = [int(i) for i in sim]
    print("WS: "+str(sim))
    sims_ws.append(sim)
    nets_ws.append(ws)
    
    #Set neural network graph layer from Scale Free Network Adjacency
    A_sf=nx.to_numpy_array(sf)  

    network_sf=tf.keras.layers.Dense(
        int(m.numpy()), use_bias=False,
        kernel_initializer=tf.compat.v1.keras.initializers.Constant(A_sf),
        trainable=False,dtype=tf.float32
    )
    
    x=[p.numpy(),q.numpy(),w0.numpy(),w1.numpy(),w2.numpy(),w3.numpy(),w4.numpy(),w5.numpy(),w6.numpy(),m.numpy()]+thresholds
    #Run Simulation
    sim,_,_=marketing_mix_cn_model(x,t,ws,initial_values,mmix,thresholds,network_sf,test=False)
    sim = [int(i) for i in sim]
    print("SF: "+str(sim))
    sims_sf.append(sim)
    nets_sf.append(sf)
    
for i,net in enumerate(nets_ws):
    nx.write_graphml(net,"net_ws_"+str(i)+".graphml")
    
for i,net in enumerate(nets_sf):
    nx.write_graphml(net,"net_sf_"+str(i)+".graphml")
