import networkx as nx
import matplotlib
import numpy as np
import random
from plots import draw_graph

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


EDGE_NUM = 914
INF = 99999


def read_data(edges):
    file_path = "data.txt"
    f_read = open(file_path, 'r', encoding='utf-8')
    for line in f_read:
        temp_list = line.replace('\n', '').split(' ')
        edges.append((int(temp_list[0]) - 1, int(temp_list[1]) - 1))
    f_read.close()


def creat_graph(G, edges):
    G.add_edges_from(edges)


# 待实现
# 1.node-degree distribution       √
# 2.average shortest path-length   √
# 3.clustering coefficient         √
# 4.graph coreness                 √
# 5.node coreness                  √
# 6.intention attack test          √
# 7.random attack test             √     

# 1.node-degree distribution
def node_degree_distribution(G):
    max_degree = 0
    degree_list = [0 for _ in range(G.number_of_nodes())]
    for i in range(G.number_of_nodes()):
        degree_list[G.degree[i]] += 1
        if G.degree[i] > max_degree:
            max_degree = G.degree[i]

    x_data = range(max_degree)
    y_data = degree_list
    plt.figure()
    for i in range(len(x_data)):
        plt.bar(x_data[i], y_data[i])
    plt.title("node-degree distribution")
    plt.xlabel("degree")
    plt.ylabel("num of nodes")
    plt.savefig("node_degree_distribution.png")

# 2.average shortest path-length
def average_shortest_path_length(G):
    adj_mat = np.array(nx.adjacency_matrix(G).todense())
    for i in range(G.number_of_nodes()):
        for j in range(G.number_of_nodes()):
            if i != j and adj_mat[i][j] == 0:
                adj_mat[i][j] = INF
    dist = list(map(lambda p: list(map(lambda q: q, p)), adj_mat.tolist()))
    # Adding vertices individually
    for r in range(G.number_of_nodes()):
        for p in range(G.number_of_nodes()):
            for q in range(G.number_of_nodes()):
                dist[p][q] = min(dist[p][q], dist[p][r] + dist[r][q])
    result = np.sum(np.array(dist)) / (G.number_of_nodes() * G.number_of_nodes() - G.number_of_nodes())
    return result

# 3.clustering coefficient
def one_node_clustering_coefficient(G, node_id):
    triangle = 0
    degree = G.degree[node_id]
    adj_list = list(G.adj[node_id])
    adj_len = len(adj_list)
    for i in range(adj_len - 1):
        for j in range(i + 1, adj_len):
            if G.has_edge(adj_list[i], adj_list[j]):
                triangle += 1
    return 2 * triangle / ( (degree) * (degree - 1) )


def clustering_coefficient(G: nx.Graph):
    cc = 0
    for node_id, degree in list(G.degree()):
        if degree >= 2:
            cc += one_node_clustering_coefficient(G, node_id)
    return cc/G.number_of_nodes()

# 4.graph coreness
def graph_coreness(G):
    H = G.copy()
    for i in range(G.number_of_nodes()):
        node_list = list(H.nodes)
        for node in node_list:
            if H.degree[node] <= i:
                H.remove_node(node)
        node_list = list(H.nodes)
        for node in node_list:
            if H.degree[node] <= i:
                H.remove_node(node)
        if nx.is_empty(H):
            break
    return i


# 5.node coreness
def node_coreness(G, node_i):
    H = G.copy()
    for i in range(G.number_of_nodes()):
        node_list = list(H.nodes)
        for node in node_list:
            if H.degree[node] <= i:
                H.remove_node(node)
        node_list = list(H.nodes)
        for node in node_list:
            if H.degree[node] <= i:
                H.remove_node(node)
        node_list = list(H.nodes)
        if node_i not in node_list:
            break
    return i


# 6.intention attack
def intention_attack(G: nx.Graph, n_attacks=3):
    n_node = G.number_of_nodes()
    # we assume each time attack the nodes which has the largest degree
    node_degrees = sorted(list(G.degree()), key=lambda x: -x[1])
    avg_shortest_paths, clustering_coeffs, g_coreness = [], [], []
    for i in range(n_attacks):
        node_id, _ = node_degrees[n_node - i - 1]
        G.remove_node(node_id)
        avg_shortest_paths.append(average_shortest_path_length(G))
        clustering_coeffs.append(clustering_coefficient(G))
        g_coreness.append(graph_coreness(G))
    draw_attack("intention_attack", n_attacks, avg_shortest_paths, clustering_coeffs, g_coreness)

# 7.random attack
def random_attack(G: nx.Graph, n_attacks=3):
    # we assume each time attack the nodes which has the largest degree
    avg_shortest_paths, clustering_coeffs, g_coreness = [], [], []
    node_degrees = list(G.degree())
    for i in range(n_attacks):
        index = random.randrange(len(node_degrees))
        node_id, _ = node_degrees.pop(index)
        G.remove_node(node_id)
        avg_shortest_paths.append(average_shortest_path_length(G))
        clustering_coeffs.append(clustering_coefficient(G))
        g_coreness.append(graph_coreness(G))

    draw_attack("random_attack", n_attacks, avg_shortest_paths, clustering_coeffs, g_coreness)


def draw_attack(mode, n_attacks, avg_shortest_paths, clustering_coeffs, g_coreness):
    plt.subplot(2, 2, 1)
    plt.plot(np.arange(n_attacks), avg_shortest_paths, color = "red") 
    plt.xticks([i for i in range(n_attacks)])   
    plt.grid()
    plt.xlabel("attack times")
    plt.title("average shortest path length")

    plt.subplot(2, 2, 2)
    plt.plot(np.arange(n_attacks), clustering_coeffs, color = "yellow")       
    plt.xticks([i for i in range(n_attacks)])   
    plt.grid()
    plt.xlabel("attack times")
    plt.title("clustering coefficient")

    plt.subplot(2, 2, 3)
    plt.plot(np.arange(n_attacks), g_coreness, color = "blue")
    plt.xticks([i for i in range(n_attacks)])   
    plt.grid()
    plt.xlabel("attack times")
    plt.title("graph coreness")

    plt.suptitle(mode)
    plt.tight_layout()
    plt.savefig("%d_%s.png" % (n_attacks, mode))



if __name__ == '__main__':
    edges = []
    read_data(edges)
    G = nx.Graph()
    creat_graph(G, edges)
    draw_graph(G)
    # node_degree_distribution(G)
    # print(average_shortest_path_length(G))
    # print(clustering_coefficient(G))
    # print(graph_coreness(G))
    # print(node_coreness(G, 8))
    intention_attack(G, n_attacks=10)
    random_attack(G, n_attacks=10)

