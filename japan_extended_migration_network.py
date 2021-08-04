import pandas as pd
import networkx as nx

#loading data
adj_prefectures = pd.read_excel("Adjacency_Prefectures.xlsx").drop('Unnamed: 0',axis=1)
mun_pre = pd.read_excel("japan_prefecture_municipality.xlsx")
G=nx.read_graphml("japan_municipalities.xml")


municipalities = list(mun_pre['Municipality'].unique())
prefectures = list(mun_pre['Prefecture'].unique())
lookup=nx.get_node_attributes(G,"name")
lookup={v: k for k, v in lookup.items()}

#corrections
adj_prefectures.rename(dict(enumerate(prefectures, start=0)),inplace=True)
mina_node_id=lookup['Minamimaki-mura (Nagano Pref.)ã']
nx.set_node_attributes(G,{mina_node_id:'Minamimaki-mura (Nagano Pref.)'},name="name")
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
            G.add_edge(lookup[mun1],lookup[mun2])

A = nx.attr_matrix(G, node_attr="name")
pd.DataFrame(A[0],index=A[1],columns=A[1]).to_excel('japan_municipalities_adj_extended.xlsx')
nx.draw(G)
nx.write_graphml(G,'japan_municipalities_extended.xml')
