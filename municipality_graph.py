import networkx as nx
from shapely.geometry import shape
import fiona
import itertools
import pandas as pd

geoms = fiona.open("jpn_admbnda_adm2_2019.shp")
G = nx.DiGraph()
municipalities=dict()
for municipality in geoms:
    G.add_node(municipality['id'],name=municipality['properties']['ADM2_EN'])
    municipalities[municipality['id']]=shape(municipality['geometry'])
for id1,id2 in itertools.combinations(municipalities.keys(), 2):
    if id1 == id2:
        continue
    poly1=municipalities[id1]
    poly2=municipalities[id2]
    if poly1.intersects(poly2):
        G.add_edge(id1,id2)
A = nx.attr_matrix(G, node_attr="name")
pd.DataFrame(A[0],index=A[1],columns=A[1]).to_excel('japan_municipalities_adj.xlsx')
nx.draw(G)
nx.write_graphml(G,'japan_municipalities.xml')
