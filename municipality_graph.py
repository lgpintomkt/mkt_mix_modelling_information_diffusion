# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 22:32:59 2021

@author: MMSETUBAL
"""

import networkx as nx
from shapely.geometry import shape
import fiona
import itertools
import pandas as pd

geoms = fiona.open("jpn_admbnda_adm2_2019.shp")
G = nx.DiGraph()
municipalities=dict()
for municipality in geoms:
    if municipality['properties']['ADM2_PCODE'] == 'JP10009':
        municipality['properties']['ADM2_EN'] = 'Minamimaki-mura (Gunma Pref.)'
    mun_name=municipality['properties']['ADM1_EN'].strip('Â\xa0').replace("HyÅgo","Hyōgo").replace("Åita","Ōita")+", "+municipality['properties']['ADM2_EN']
    G.add_node(municipality['id'],name=mun_name)
    municipalities[municipality['id']]=shape(municipality['geometry'])
for id1,id2 in itertools.combinations(municipalities.keys(), 2):
    if id1 == id2:
        continue
    poly1=municipalities[id1]
    poly2=municipalities[id2]
    if poly1.intersects(poly2):
        G.add_edge(id1,id2)
        G.add_edge(id2,id1)
#correction, Ishioka-machi in Fukushima and Tono-cho & Kadokawa-cho in Miyazaki doesn't seem to exist
G.remove_node('407')
G.remove_node('1803')
G.remove_node('1804')
A = nx.attr_matrix(G, node_attr="name")
pd.DataFrame(A[0],index=A[1],columns=A[1]).to_excel('japan_municipalities_adj.xlsx')
#nx.draw(G)
nx.write_graphml(G,'japan_municipalities.xml')
