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
import copy
from numba import jit
from scipy.optimize import minimize, differential_evolution
from scipy.optimize import approx_fprime
import requests
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
gpu=tf.config.list_physical_devices('GPU')[0]
tf.config.set_logical_device_configuration(gpu,[tf.config.LogicalDeviceConfiguration(memory_limit=17000)])

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
                baseline=np.sum(threshold_model(t,graph,*args[1:])[1])
            print("Influencer candidate "+str(candidate)+": "+node_name)
            #pdb.set_trace()
            new=np.sum(threshold_model(t,graph,*args[1:])[1])
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
G=nx.read_graphml(data_path+"\\Data\\Network\\Network Files\\japan_municipalities_extended_20220920.xml")

L=19 #real L is +1 since it doesn't include special influencer class

population=nx.get_node_attributes(G,'population')
total_pop = 125621730 #source: https://www.worldometers.info/world-population/japan-population/
pop_norm=dict()
#to impute missing values
mean_pop=(total_pop/len(G.nodes()))/total_pop
for node,pop in zip(population.keys(),population.values()):
    pop_norm[node]=pop/total_pop
    
for node in G.nodes():
    try:
        pop_norm[node]
    #missing cases
    except:
        pop_norm[node]=mean_pop

nx.set_node_attributes(G,pop_norm,name='normalized population')
pop_norm = tf.convert_to_tensor(np.array(list(nx.get_node_attributes(G,'normalized population').values())))

random.seed(5718)
np.random.seed(5718)
link_class=link_classes(G,L)
degrees=G.out_degree(weight='weight')
link_class_thres=dict()
link_class_thres[0]=random.uniform(1e8,1.5e8)
for link in list(link_class.keys())[1:]:
    link_class_thres[link]=random.uniform(0,7e7)
    if link_class_thres[link]>link_class_thres[link-1]:
        link_class_thres[link]=random.uniform(0,link_class_thres[link-1])
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

m=18000000
adoption=np.array(data['Adoption'])
t=np.array(data.index)
t_train=np.array(data.index)[:50]
t_val=np.array(data.index)[50:60]
t_test=np.array(data.index)[60:]
mmix=np.array(data[['Product','Price','Place','Promotion']])

p=tf.Variable(5e-3,trainable=True,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
q=tf.Variable(5e-5,trainable=True,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)
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
m=tf.Variable(3e8,trainable=True,dtype=tf.float32,constraint=tf.keras.constraints.NonNeg)

bc=11197800

influencers=centrality_heuristic(G,k=5,method='eigenvector',output="id")+centrality_heuristic(G,k=5,method='outdegree',output="id")
influencers=[int(i) for i in influencers]

A=nx.to_numpy_array(G,weight='weight')  

#remove weights
A[A>0]=1


network=tf.keras.layers.Dense(
    1889, use_bias=False,
    kernel_initializer=tf.compat.v1.keras.initializers.Constant(A),
    trainable=True,dtype=tf.float32
)

def marketing_mix_cn_model(x,steps,G,bc,mmix,thres,prev_pred=None,test=False):
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
        thresholds=dict()
        for node in G.nodes():
            for index,link in enumerate(thres):
                if link_classes[node] == str(index):
                    thresholds[node]=thres[index]
        state=bc
    f_t=np.zeros([1889,73])
    f_t[:,0]=tf.nn.relu(state)
    
    
    pred=np.zeros(len(steps))

    if prev_pred is None:
        prev_pred=tf.cast(bc.sum(),tf.float32)

    potential=m*tf.cast(pop_norm,tf.float32)
    activation=np.zeros(73)

    t_0=steps[0]
    
    for t in steps:
        U=np.exp(-t/w4)*tfp.math.trapz(np.exp(-steps[0:t-t_0]/w4)*np.divide(np.exp(steps[0:t-t_0]/w4)*f_t[:,0:t-t_0]*mmix[0:t-t_0,0],w4))
        S=network(tf.reshape(state+tf.nn.relu(tf.cast(q*tf.cast(U,tf.float32),tf.float32)),[1,1889]))
        R=w0+p*(-w1)*(mmix[t-t_0,1]+1e-12)*(m*w2*mmix[t-t_0,2]-prev_pred)*w3*(mmix[t-t_0,3]+1e-12)
        f_t[:,t-t_0]=tf.round(potential*tf.nn.relu(tf.nn.tanh(S+R)))

        state=f_t[:,t-t_0]
        
        adopt=tf.nn.relu(state)
        pred[t-t_0]=tf.reduce_sum(tf.nn.relu(adopt))
        activation[t-t_0]=tf.reduce_sum(tf.cast(tf.math.greater(adopt, 0.), tf.float32))
        if test:
            tf.print("Step: ", t, " Adoption:", pred[t-t_0], " Activation:", activation[t-t_0])
    return pred,state

def marketing_mix_cn_validation(x):
    global L
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
    thresholds=[]
    
    for i in range(L):
        thresholds.append(x[10+i])

    global t_train
    global G
    global bc_v
    global mmix
    global adoption
    global initial_values
    global iteration
    global batch
    
    loss=tf.reduce_sum((adoption[50:60] - marketing_mix_cn_model(x,t_val,G,bc_v,mmix,thresholds,initial_values)[0])**2)
    tf.print("Iteration: "+str(iteration)+" Batch: "+str(batch)+" Validation Loss: ",loss)
    return loss.numpy()

def marketing_mix_cn_residuals(x):
    global L
    pos=x

    p.assign(pos[0])
    q.assign(pos[1])
    w0.assign(pos[2])
    w1.assign(pos[3])
    w2.assign(pos[4])
    w3.assign(pos[5])
    w4.assign(pos[6])
    w5.assign(pos[7])
    w6.assign(pos[8])
    m.assign(pos[9])
    thresholds=[]
    for i in range(L):
        thresholds.append(pos[10+i])
    
    global iteration
    global t_train
    global G
    global bc_v
    global mmix
    global adoption
    global initial_values
    global batch
    
    pos_loss=tf.reduce_sum((adoption[0:50] - marketing_mix_cn_model(pos,t_train,G,bc_v,mmix,thresholds,initial_values)[0])**2)
    
    tf.print("Iteration: "+str(iteration)+" Batch: "+str(batch)+" Training Loss: ",pos_loss)
    batch+=1
    return pos_loss.numpy()

bc_v = np.zeros([1889])
initial_values=np.zeros([1889])
for influencer in influencers:
    bc_v[influencer]=pop_norm[influencer]
bc_v=bc_v/bc_v.sum()
bc_v=bc*bc_v

val=[]
iteration=1
batch=1
min_vals=1
for i in range(3000):
    if i==0:
        marketing_mix_cn_diffusion=tfp.optimizer.nelder_mead_minimize(
            objective_function=marketing_mix_cn_residuals,
            initial_vertex=tf.constant(list(np.random.rand(9))+[2e7]+thresholds,dtype=tf.float32),
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
        if val[-1]>val[-2] and len(val)>=min_vals:
            print("Early Stopping Condition Met.")
            break
    iteration+=1
    
print("In-Sample Forecast")
x=marketing_mix_cn_diffusion.position
marketing_mix_cn_forecast_is,intermediate_state=marketing_mix_cn_model(x,t_train,G,bc_v,mmix,thresholds,test=True)
marketing_mix_cn_forecast_val,final_state=marketing_mix_cn_model(x,t_val,G,intermediate_state,mmix,thresholds,prev_pred=marketing_mix_cn_forecast_is[-1],test=True)
marketing_mix_cn_forecast_oos,_=marketing_mix_cn_model(x,t_test,G,final_state,mmix,thresholds,prev_pred=marketing_mix_cn_forecast_val[-1],test=True)

#Final influencer discovery using MMD model
influencers=kempe_greedy_algorithm(G,marketing_mix_cn_model,t_train,G,bc_v,mmix,thresholds,initial_values)

bc_v = np.zeros([1889])
initial_values=np.zeros([1889,1889])
for influencer in influencers:
    bc_v[influencer]=pop_norm[influencer]
bc_v=bc_v/bc_v.sum()
bc_v=bc*bc_v
for influencer_1 in influencers:
    for influencer_2 in influencers:
        initial_values[influencer_1,influencer_2]=bc_v[influencer_1]

val=[]
iteration=1
batch=1
min_vals=1
for i in range(3000):
    if i==0:
        marketing_mix_cn_diffusion=tfp.optimizer.nelder_mead_minimize(
            objective_function=marketing_mix_cn_residuals,
            initial_vertex=tf.constant(list(np.random.rand(10))+[2e7]+thresholds,dtype=tf.float32),
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
        if val[-1]>val[-2] and len(val)>=min_vals:
            print("Early Stopping Condition Met.")
            break
    iteration+=1
    
print("In-Sample Forecast")
x=marketing_mix_cn_diffusion.position
marketing_mix_cn_forecast_is,intermediate_state=marketing_mix_cn_model(x,t_train,G,bc_v,mmix,thresholds,test=True)
marketing_mix_cn_forecast_val,final_state=marketing_mix_cn_model(x,t_val,G,intermediate_state,mmix,thresholds,prev_pred=marketing_mix_cn_forecast_is[-1],test=True)
marketing_mix_cn_forecast_oos,_=marketing_mix_cn_model(x,t_test,G,final_state,mmix,thresholds,prev_pred=marketing_mix_cn_forecast_val[-1],test=True)
