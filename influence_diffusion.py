# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 16:01:59 2021

@author: MMSETUBAL
"""

import networkx as nx
from rpy2 import robjects
import copy
import random
import pdb
import operator
import rpy2.robjects.packages as rpackages

igraph = rpackages.importr("igraph")
base = rpackages.importr("base")

def linear_threshold_model(G,seeds,steps=10):
    actives=copy.deepcopy(seeds)
    name_dict=nx.get_node_attributes(G,"name")
    thresholds=nx.get_node_attributes(G,"threshold")
    for step in range(steps):
        for node_i in G.nodes:
            if name_dict[node_i] in actives:
                continue
            influence_function=1e-20
            for node_j in G.predecessors(node_i):
                if name_dict[node_j] in actives:
                    influence_function+=G.get_edge_data(node_j,node_i)['weight']
            if influence_function>thresholds[node_i]:
                 actives.append(name_dict[node_i])
    return actives

def continuous_threshold_model(G,seeds,steps=10):
    actives=copy.deepcopy(seeds)
    name_dict=nx.get_node_attributes(G,"name")
    thresholds=nx.get_node_attributes(G,"threshold")
    influence_functions=dict()
    for node in G.nodes:
        influence_functions[node]=1e-20
    for step in range(steps):
        for node_i in G.nodes:
            if name_dict[node_i] in actives:
                continue
            influence_function=1e-20
            for node_j in G.predecessors(node_i):
                if name_dict[node_j] in actives:
                    influence_function+=G.get_edge_data(node_j,node_i)['weight']*influence_functions[node_j]
            influence_functions[node_i]=influence_function
        for node_i in G.nodes:
            if influence_functions[node_i]>thresholds[node_i]:
                actives.append(name_dict[node_i])
    return actives,influence_functions

def kempe_greedy_algorithm(G,threshold_model,k=10,steps=10):
    names=nx.get_node_attributes(G,"name")
    graph=copy.deepcopy(G)
    influencers=[]
    for i in range(k):
        activation=dict()
        for node in graph.nodes():
            node_name=names[node]
            activation[node]=sum(continuous_threshold_model(graph,influencers+[node_name],steps)[1].values())-sum(continuous_threshold_model(graph,influencers,steps)[1].values())
            #print("k="+str(k)+" "+" n="+node+" ("+node_name+"): "+str(activation[node]))
        top_influencer=max(activation.items(), key=operator.itemgetter(1))[0]
        influencers.append(names[top_influencer])
        graph.remove_node(top_influencer)
    return influencers
        
def collective_influence(graph_path,data_path,script_path,k=10):
    base.source(script_path)
    robjects.r("library(igraph)")
    robjects.r(r"g<-read_graph('"+graph_path+"',format='graphml')")
    robjects.r("g<-delete_vertices(g,102)")
    robjects.r("influencers<-getInfluencers(g,d=3,maxInfluencers="+str(k)+")")
    influencers=list(robjects.r['influencers'][1])
    return influencers

def centrality_heuristic(G,method='outdegree',k=10):
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
        influencers.append(names[top_influencer])
        del centralities[str(top_influencer)]
    return influencers

graph_path=r"C:\\Users\\MMSETUBAL\\Desktop\\Artigos\\MMM and Information Diffusion over Complex Networks\\Data\\Network\\Network Files\\japan_municipalities.xml"
data_path="C:\\Users\\MMSETUBAL\\Desktop\\Artigos\\MMM and Information Diffusion over Complex Networks"
script_path=r"C:\Users\MMSETUBAL\Desktop\Artigos\MMM and Information Diffusion over Complex Networks\Progs\R\collective_influence_algorithm.R"    

G=nx.read_graphml(data_path+"\\Data\\Network\\Network Files\\japan_municipalities_extended.xml")
nodes=nx.get_node_attributes(G,"name")
thresholds=dict()
for node in G.nodes():
    thresholds[node]={'threshold':random.uniform(0,1)}
nx.set_node_attributes(G, thresholds)

influencers_ci=collective_influence(graph_path,data_path,script_path,k=10)

influencers_degree=centrality_heuristic(G,method='outdegree',k=10)

influencers_eigenvector=centrality_heuristic(G,method='eigenvector',k=10)

influencers_closeness=centrality_heuristic(G,method='closeness',k=10)

influencers_kempe=kempe_greedy_algorithm(G,continuous_threshold_model,k=10)


