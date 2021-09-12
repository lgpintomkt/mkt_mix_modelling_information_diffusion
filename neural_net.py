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
from scipy.optimize import approx_fprime
import requests

gpu=tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_logical_device_configuration(gpu,[tf.config.LogicalDeviceConfiguration(memory_limit=1700)])

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

random.seed(32466)
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

data=pd.read_csv(data_path+"\\Data\\Statistical\\mmm_data.csv",sep=";").set_index("t")
#m=data['Adoption'].iloc[-1]
m=18000000
adoption=np.array(data['Adoption'])
t=np.array(data.index)
t_train=np.array(data.index)[:50]
t_val=np.array(data.index)[50:60]
t_test=np.array(data.index)[60:]
mmix=np.array(data[['Product','Price','Place','Promotion']])

p=tf.Variable(5e-3,trainable=True,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
q=tf.Variable(5e-3,trainable=True,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
w0=tf.Variable(5e-3,trainable=True,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
w1=tf.Variable(5e-3,trainable=True,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
w2=tf.Variable(5e-3,trainable=True,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
w3=tf.Variable(5e-3,trainable=True,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
w4=tf.Variable(1,trainable=True,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
w5=tf.Variable(1,trainable=True,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
w6=tf.Variable(0.5,trainable=True,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
w7=tf.Variable(2,trainable=True,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
w8=tf.Variable(1e-3,trainable=False,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
lamb=tf.Variable(2,trainable=True,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
m=tf.Variable(2e7,trainable=True,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)



bc=11197800

influencers=centrality_heuristic(G,k=5,method='eigenvector',output="id")+centrality_heuristic(G,k=5,method='outdegree',output="id")
influencers=[int(i) for i in influencers]

initial_values=np.zeros([1889,1889])
for influencer in influencers:
    initial_values[influencer,:]=(bc/len(influencers))

A=nx.to_numpy_array(G,weight='normalized weight')  
network=tf.keras.layers.Dense(
    1889, use_bias=False,
    kernel_initializer=tf.compat.v1.keras.initializers.Constant(A),
    trainable=False,dtype=tf.float32
)




def submodular_threshold_model(t,G,bc,m,mmix,thres,influencers,clipval=1e4):

    global network    
    global link_classes
    thresholds=dict()
    for node in G.nodes():
        for index,link in enumerate(thres):
            if link_classes[node] == str(index):
                thresholds[node]=thres[index]
    thres_var=tf.Variable(np.tile(np.array(list(thresholds.values()))*-1,[1889,1]),trainable=False,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
    
    maximum_radiation=tf.math.reduce_max(w0+p*(-w1)*((mmix[:,1]+1e-12)**(-lamb))*(m*w2*mmix[:,2]-bc)*w3*np.sqrt(mmix[:,3]))
    values=tf.clip_by_value(tf.nn.relu(network(influencers)+(thres_var-maximum_radiation)),0,clipval)
    u_0=w5*np.exp(-1/w4)
    f_i=np.zeros([1889,1889,73])
    f_i[:,:,0]=tf.multiply(np.exp(1/w4),values)+w5*np.exp(-1/w4)
    u=np.zeros([1889,73])
    u[:,0]=u_0
    adopt=np.zeros([1889,72])
    pred=np.zeros(len(t))
    activation=np.zeros(73)
    activation[0]=tf.reduce_sum(tf.cast(tf.math.greater(values[:,0], 0.), tf.float32))
    
    for i in t:
        radiation=1
        f_i[:,:,i]=tf.multiply(np.exp(i/w4),values)+w5*np.exp(-i/w4)
        if i==1:
            u[:]=w5*np.exp(-t[0]/w4)
        else:
            u[:]=w6*np.exp(-t[i-1]/w4)*tf.cast(tfp.math.trapz(np.exp(-t[0:i-1]/w4)*np.divide(np.exp(t[0:i-1]/w4)*tf.reduce_sum(tf.nn.relu(f_i[:,:,0:i-1]),axis=1)*mmix[0:i-1,0],w4)),dtype=tf.float32)+w5*np.exp(-t[i-1]/w4)
        diffusion=q*u[:]
        adopt[:,i-1]=radiation+diffusion
        pred[i-1]=tf.reduce_sum(adopt[:,i-1])*tf.math.exp(-1/72)
        values=tf.clip_by_value(tf.nn.relu(tf.transpose(network(values))),0,clipval)
        activation[i]=tf.reduce_sum(tf.cast(tf.math.greater(values[:,i-1], 0.), tf.float32))
        #tf.print("Step: ", i, " Adoption:", pred[i-1], " Activation:", tf.math.reduce_max(activation))

    return pred

def marketing_mix_cn_model(t,G,bc,m,mmix,thres,influencers,clipval=1e1,test=False):

    global network    
    global link_classes
    thresholds=dict()
    for node in G.nodes():
        for index,link in enumerate(thres):
            if link_classes[node] == str(index):
                thresholds[node]=thres[index]
    thres_var=tf.Variable(np.tile(np.array(list(thresholds.values()))*-1,[1889,1]),trainable=False,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
    
    values=network(influencers)+thres_var
    f_i=np.zeros([1889,1889,73])
    f_i[:,:,0]=tf.multiply(np.exp(1/w4),values)+w5*np.exp(-1/w4)
    #import pdb;pdb.set_trace()
    u=np.zeros([1889])
    pred=np.zeros(len(t))
    activation=np.zeros(73)
    for i,ti in enumerate(t):
        radiation=tf.keras.activations.sigmoid(w0+p*(-w1)*(mmix[ti,1]+1e-12)*(m*w2*mmix[ti,2]-bc)*w3*(mmix[ti,3]+1e-12))
        f_i[:,:,ti]=tf.multiply(np.exp(ti/w4),tf.nn.relu(tf.keras.activations.tanh(values)))+w5*np.exp(-ti/w4)
        if i==1:
            u[:]=w5*np.exp(-t[0]/w4)
        else:
            u[:]=w6*np.exp(-t[i]/w4)*tf.cast(tfp.math.trapz(np.exp(-t[0:i]/w4)*np.divide(np.exp(t[0:i]/w4)*tf.reduce_sum(tf.nn.relu(f_i[:,:,0:i]),axis=1)*mmix[0:i,0],w4)),dtype=tf.float32)+w5*np.exp(-t[i]/w4)
        diffusion=q*u[:]
        #import pdb;pdb.set_trace()
        pred[i]=bc+(m-bc)*(radiation+tf.reduce_mean(diffusion))/2
        pred[i]=tf.reduce_max(pred)
        values=network(tf.nn.relu(tf.transpose(values)))
        activation[i]=tf.reduce_max(tf.reduce_sum(tf.cast(tf.math.greater(values, 0.), tf.float32),axis=1))
        if test:
            tf.print("Step: ", ti, " Adoption:", pred[i], " Activation:", tf.reduce_max(activation))
        #tf.print("Step: ", ti, " Adoption:", pred[i], " Activation:", tf.reduce_max(activation))
    return pred

def marketing_mix_cn_validation(x):
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
    #w7.assign(x[9])
    #w7.assign(x[10])
    thresholds=[]
    for i in range(10):
        thresholds.append(x[10+i])
    
    global t_train
    global G
    global bc
    global mmix
    global adoption
    global initial_values
    global iteration
    global batch
    loss=tf.reduce_sum((adoption[50:60] - marketing_mix_cn_model(t_val,G,bc,m,mmix,thresholds,initial_values))**2)
    tf.print("Iteration: "+str(iteration)+" Batch: "+str(batch)+" Validation Loss: ",loss)
    return loss.numpy()

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
    m.assign(x[9])
    #w7.assign(x[9])
    #w8.assign(x[10])
    thresholds=[]
    for i in range(10):
        thresholds.append(x[10+i])
    
    global iteration
    global t_train
    global G
    global bc
    global mmix
    global adoption
    global initial_values
    global batch
    loss=tf.reduce_sum((adoption[0:50] - marketing_mix_cn_model(t_train,G,bc,m,mmix,thresholds,initial_values))**2)
    tf.print("Iteration: "+str(iteration)+" Batch: "+str(batch)+" Training Loss: ",loss)
    batch+=1
    return loss.numpy()

val=[]
iteration=1
batch=1
min_vals=5
for i in range(300):
    if i==0:
        marketing_mix_cn_diffusion=tfp.optimizer.nelder_mead_minimize(
            objective_function=marketing_mix_cn_residuals,
            initial_vertex=tf.constant([5e-2,5e-2,5e-3,5e-3,5e-3,5e-3,1,1,0.5,2e7]+thresholds), 
            max_iterations=tf.constant(1)
            )
    else:
        marketing_mix_cn_diffusion=tfp.optimizer.nelder_mead_minimize(
            objective_function=marketing_mix_cn_residuals,
            initial_vertex=x, 
            max_iterations=tf.constant(1)
            )
    batch=1
    x=marketing_mix_cn_diffusion.position
    val.append(marketing_mix_cn_validation(x))
    if len(val)>1:
        if val[-1]>val[-2] and len(val)>=5:
            print("Early Stopping Condition Met.")
            break
    iteration+=1
    
print("In-Sample Forecast")
x=marketing_mix_cn_diffusion.position
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
#w7.assign(x[9])
#w8.assign(1e-3)
thresholds=[]
for i in range(10):
    thresholds.append(x[10+i])
        
    
marketing_mix_cn_forecast_is=marketing_mix_cn_model(t_train,G,bc,m,mmix,thresholds,initial_values,test=True)

influencers=kempe_greedy_algorithm(G,submodular_marketing_threshold_model,t,G,bc,m,mmix,thresholds,initial_values)
