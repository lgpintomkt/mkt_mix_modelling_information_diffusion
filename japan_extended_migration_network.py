# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 02:43:54 2021

@author: MMSETUBAL
"""

import pandas as pd
import networkx as nx

#loading data
adj_prefectures = pd.read_excel("Adjacency_Prefectures.xlsx").drop('Unnamed: 0',axis=1)
mun_pre = pd.read_excel("japan_prefecture_municipality.xlsx")
G=nx.read_graphml("japan_municipalities.xml")
pop = pd.read_excel("japan_population.xlsx").drop_duplicates(subset="Name",keep="first").set_index("Name").to_dict(orient='index')
#5 duplicated municipalities were found, but these are mostly due to different conventions
#since it wasn't possible to determine the most accurate population value
#we chose to keep the first, since the numerical pop. values for the duplicated municipalities
#were very close, and the number of repeated cases was small (5 out of 1888)
#which meant that it will not significantly change the results locally (edge weight) or globally

municipalities = list(mun_pre['Municipality'].unique())
prefectures = list(mun_pre['Prefecture'].unique())
lookup=nx.get_node_attributes(G,"name")
lookup={v: k for k, v in lookup.items()}

#corrections
adj_prefectures.rename(dict(enumerate(prefectures, start=0)),inplace=True)
mina_node_id=lookup['Nagano, Minamimaki-mura (Nagano Pref.)ã']
nx.set_node_attributes(G,{mina_node_id:'Nagano, Minamimaki-mura (Nagano Pref.)'},name="name")
lookup=nx.get_node_attributes(G,"name")
lookup={v: k for k, v in lookup.items()}

prefs=list(adj_prefectures.columns)

for i,j in zip(prefectures,prefs):
    if i!=j:
        print(i+' '+j)
#prefectures name match between datasets

prefecture_mun=dict()

for pref in prefectures:
    muns=list(mun_pre[mun_pre['Prefecture']==pref]['Municipality'])
    for mun in muns:
        prefecture_mun[mun]=pref

for mun1 in municipalities:
    for mun2 in municipalities:
        if adj_prefectures.at[prefecture_mun[mun1],prefecture_mun[mun2]] == 1:
            mun1_name=prefecture_mun[mun1]+", "+mun1
            mun2_name=prefecture_mun[mun2]+", "+mun2
            if mun1_name == 'Chiba, Ychiyo City':
                pop[mun1_name]={'Population': 193152} #source https://www.citypopulation.de/en/japan/chiba/
            if mun2_name == 'Chiba, Ychiyo City':
                pop[mun2_name]={'Population': 193152} #source https://www.citypopulation.de/en/japan/chiba/
            if mun1_name == 'Hokkaido, Kawakami-cho':
                pop[mun1_name]={'Population': 4044} #source https://www.citypopulation.de/en/japan/hokkaido/
            if mun2_name == 'Hokkaido, Kawakami-cho':
                pop[mun2_name]={'Population': 4044} #source https://www.citypopulation.de/en/japan/hokkaido/
            if mun1_name == 'Yamaguchi, Shinomoseki City':
                pop[mun1_name]={'Population': 268517} #source https://www.citypopulation.de/en/japan/yamaguchi/_/35201__shimonoseki/
            if mun2_name == 'Yamaguchi, Shinomoseki City':
                pop[mun2_name]={'Population': 268517} #source https://www.citypopulation.de/en/japan/yamaguchi/_/35201__shimonoseki/
            if mun1_name == 'Yamaguchi, Suo-Oshima-cho':
                pop[mun1_name]={'Population': 17199} #source https://www.citypopulation.de/en/japan/yamaguchi/%C5%8Dshima/35305__su%C5%8D_%C5%8Dshima/
            if mun2_name == 'Yamaguchi, Suo-Oshima-cho':
                pop[mun2_name]={'Population': 17199} #source https://www.citypopulation.de/en/japan/yamaguchi/%C5%8Dshima/35305__su%C5%8D_%C5%8Dshima/
            if mun1_name == 'Fukushima, Ishioka-machi' or mun2_name == 'Fukushima, Ishioka-machi':
                continue
            if mun1_name == 'Miyazaki, Tono-cho' or mun2_name == 'Miyazaki, Tono-cho':
                continue
            if mun1_name == 'Miyazaki, Kadokawa-cho' or mun2_name == 'Miyazaki, Kadokawa-cho':
                continue
            G.add_edge(lookup[mun1_name],lookup[mun2_name],weight=pop[mun1_name]['Population']/pop[mun2_name]['Population'])

A = nx.attr_matrix(G, node_attr="name")
pd.DataFrame(A[0],index=A[1],columns=A[1]).to_excel('japan_municipalities_adj_extended.xlsx')
#nx.draw(G)
nx.write_graphml(G,'japan_municipalities_extended.xml')
