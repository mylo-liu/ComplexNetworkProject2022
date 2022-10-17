'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-12-23 18:53:12
LastEditors: ZhangHongYu
LastEditTime: 2022-04-25 15:13:26
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx
import numpy as np
# Standard Library
import random
from collections import defaultdict
from copy import copy
import os

# Third Party
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib import cm
from scipy.interpolate import splprep, splev
from scipy.spatial import ConvexHull
import pandas as pd


def _inter_community_edges(G, partition):
    edges = defaultdict(list)

    for (i, j) in G.edges():
        c_i = partition[i]
        c_j = partition[j]

        if c_i == c_j:
            continue

        edges[(c_i, c_j)].append((i, j))

    return edges


def _position_communities(G, partition, **kwargs):
    hypergraph = nx.Graph()
    hypergraph.add_nodes_from(set(partition))

    inter_community_edges = _inter_community_edges(G, partition)
    for (c_i, c_j), edges in inter_community_edges.items():
        hypergraph.add_edge(c_i, c_j, weight=len(edges))

    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # Set node positions to positions of its community
    pos = dict()
    for node, community in enumerate(partition):
        pos[node] = pos_communities[community]

    return pos


def _position_nodes(G, partition, **kwargs):
    communities = defaultdict(list)
    for node, community in enumerate(partition):
        communities[community].append(node)

    pos = dict()
    for c_i, nodes in communities.items():
        subgraph = G.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos


# Adapted from: https://stackoverflow.com/questions/43541376/how-to-draw-communities-with-networkx
def community_layout(G, partition):
    pos_communities = _position_communities(G, partition, scale=7.0)  # 10.0
    pos_nodes = _position_nodes(G, partition, scale=2.0) # 2.0

    # Combine positions
    pos = dict()
    for node in G.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos


##################
# DRAW COMMUNITIES
##################


def draw_graph(G: nx.Graph, dark=False, seed=1):

    np.random.seed(seed)
    random.seed(seed)

    communities = [[ node_id - 1 for node_id in G.nodes()]]
    partition = [0 for _ in range(G.number_of_nodes())]
    for c_i, nodes in enumerate(communities):
        for node_id in nodes:
            partition[node_id] = c_i 

    plt.rcParams["figure.facecolor"] = "black" if dark else "white"
    plt.rcParams["axes.facecolor"] = "black" if dark else "white"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")

    node_size = 10200 / G.number_of_nodes()
    linewidths = 34 / G.number_of_nodes() #34

    pos = community_layout(G, partition)
    nodes = nx.draw_networkx_nodes(
        G,
        pos=pos,
        node_color=partition,
        linewidths=linewidths,
        cmap=cm.jet,
        ax=ax,
        node_size=node_size
    )
    nodes.set_edgecolor("w")
    edges = nx.draw_networkx_edges(
        G,
        pos=pos,
        edge_color=(1.0, 1.0, 1.0, 0.75) if dark else (0.6, 0.6, 0.6, 1.0),
        width=linewidths,
        ax=ax
    )
    nx.draw_networkx_labels(G, pos=pos, ax=ax, font_color="white", font_size=node_size*0.02)


    plt.savefig("graph.png")
    
    






