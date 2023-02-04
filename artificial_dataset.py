# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 10:47:58 2022

@author: MMSETUBAL
"""

from pyDOE import *
import networkx as nx
import random

#Variables
rho=tf.Variable(5e-3,trainable=False,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
p=tf.Variable(5e-3,trainable=False,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
q=tf.Variable(5e-5,trainable=False,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
w0=tf.Variable(0,trainable=False,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
w1=tf.Variable(1,trainable=False,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
w2=tf.Variable(1,trainable=False,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
w3=tf.Variable(1,trainable=False,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
w6=tf.Variable(1,trainable=False,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)

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
#Network global var
network=tf.keras.layers.Dense(
    int(m.numpy()), use_bias=False,
    kernel_initializer=tf.compat.v1.keras.initializers.Constant(A),
    trainable=False,dtype=tf.float32
)

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

nets=[]
sims=[]
for sample in samples:
    rho.assign(sample[0])
    p.assign(sample[1])
    q.assign(sample[2]*1e-2)
    w0.assign(sample[3])
    w1.assign(sample[4]*1e-2)
    w2.assign(sample[5])
    w3.assign(sample[6])
    w6.assign(sample[7])

    #Generate Watts Strogatz Network
    ws=nx.watts_strogatz_graph(int(m.numpy()), K, rho.numpy())
    
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
    A=nx.to_numpy_array(ws)  

    network=tf.keras.layers.Dense(
        int(m.numpy()), use_bias=False,
        kernel_initializer=tf.compat.v1.keras.initializers.Constant(A),
        trainable=False,dtype=tf.float32
    )
    
    x=[p.numpy(),q.numpy(),w0.numpy(),w1.numpy(),w2.numpy(),w3.numpy(),w4.numpy(),w5.numpy(),w6.numpy(),m.numpy()]+thresholds
    #Run Simulation
    sim,_,_=marketing_mix_cn_model(x,t,ws,initial_values,mmix,thresholds,test=False)
    sim = [int(i) for i in sim]
    print(sim)
    sims.append(sim)
    nets.append(ws)
    
for i,net in enumerate(nets):
    nx.write_graphml(net,"net_"+str(i)+".graphml")
