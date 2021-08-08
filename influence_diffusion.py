# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 16:01:59 2021

@author: MMSETUBAL
"""

import networkx as nx
from rpy2 import robjects
import copy
import rpy2.robjects.packages as rpackages

igraph = rpackages.importr("igraph")
base = rpackages.importr("base")

def linear_threshold_model(G,seeds,name_dict,steps=10,threshold=3):
    actives=copy.deepcopy(seeds)
    for step in range(steps):
        for node_i in G.nodes:
            if name_dict[node_i] in actives:
                continue
            influence_function=0
            for node_j in G.predecessors(node_i):
                if name_dict[node_j] in actives:
                    influence_function+=G.get_edge_data(node_j,node_i)['weight']
            if influence_function>threshold:
                 actives.append(name_dict[node_i])
    return actives

def continuous_threshold_model(G,seeds,name_dict,steps=10,threshold=3):
    actives=copy.deepcopy(seeds)
    influence_functions=dict()
    for node in G.nodes:
        if name_dict[node] in actives:
            influence_functions[node]=1
        influence_functions[node]=0
    for step in range(steps):
        for node_i in G.nodes:
            if name_dict[node_i] in actives:
                continue
            for node_j in G.predecessors(node_i):
                if name_dict[node_j] in actives:
                    influence_functions[node_i]+=G.get_edge_data(node_j,node_i)['weight']*influence_functions[node_j]
            if influence_functions[node_i]>threshold:
                actives.append(name_dict[node_i])
    return actives

data_path="C:\\Users\\MMSETUBAL\\Desktop\\Artigos\\MMM and Information Diffusion over Complex Networks"

base.source(r"C:\Users\MMSETUBAL\Desktop\Artigos\MMM and Information Diffusion over Complex Networks\Progs\R\collective_influence_algorithm.R")
robjects.r("library(igraph)")

robjects.r(r"g<-read_graph('C:\\Users\\MMSETUBAL\\Desktop\\Artigos\\MMM and Information Diffusion over Complex Networks\\Data\\Network\\Network Files\\japan_municipalities.xml',format='graphml')")
robjects.r("g<-delete_vertices(g,102)")
robjects.r("influencers<-getInfluencers(g,d=3,maxInfluencers=10)")
influencers=list(robjects.r['influencers'][1])

G=nx.read_graphml(data_path+"\\Data\\Network\\Network Files\\japan_municipalities_extended.xml")
nodes=nx.get_node_attributes(G,"name")
# degree_centralities=G.out_degree(weight='weight')
# named_degree_centralities=dict()
# for node in nodes.keys():
#     named_degree_centralities[nodes[node]]=degree_centralities[node]

experiments = []
threshold=1e-20
for i in range(1,5):
    steps=i
    actives_linear=linear_threshold_model(G,influencers,nodes,threshold=threshold,steps=steps)
    actives_continuous=continuous_threshold_model(G,influencers,nodes,threshold=threshold,steps=steps)
    experiments.append([len(actives_linear),len(actives_continuous)])
