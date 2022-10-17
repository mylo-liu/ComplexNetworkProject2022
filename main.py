import networkx as nx
import matplotlib
import numpy as np

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

NODE_NUM = 379
EDGE_NUM = 914
INF = 99999


def read_data(edges):
    file_path = "data.txt"
    f_read = open(file_path, 'r', encoding='utf-8')
    for line in f_read:
        temp_list = line.replace('\n', '').split(' ')
        edges.append((int(temp_list[0]), int(temp_list[1])))
    f_read.close()


def creat_graph(G, edges):
    G.add_edges_from(edges)


# 画图需要改进
def draw_graph(G):
    nx.draw_networkx(G)
    plt.show()


# 待实现
# 1.node-degree distribution       √
# 2.average shortest path-length   √
# 3.clustering coefficient         √
# 4.graph coreness                 √
# 5.node coreness                  √
# 6.attention attack test
# 7.attention attack test

# 1.node-degree distribution
def node_degree_distribution(G):
    max_degree = 0
    degree_list = [0 for _ in range(NODE_NUM + 1)]
    for i in range(1, NODE_NUM + 1):
        degree_list[G.degree[i]] += 1
        if G.degree[i] > max_degree:
            max_degree = G.degree[i]

    x_data = range(max_degree + 1)
    y_data = degree_list
    for i in range(len(x_data)):
        plt.bar(x_data[i], y_data[i])
    plt.title("node-degree distribution")
    plt.xlabel("degree")
    plt.ylabel("num of nodes")
    plt.show()

# 2.average shortest path-length
def average_shortest_path_length(G):
    adj_mat = np.array(nx.adjacency_matrix(G).todense())
    for i in range(NODE_NUM):
        for j in range(NODE_NUM):
            if i != j and adj_mat[i][j] == 0:
                adj_mat[i][j] = INF
    dist = list(map(lambda p: list(map(lambda q: q, p)), adj_mat.tolist()))
    # Adding vertices individually
    for r in range(NODE_NUM):
        for p in range(NODE_NUM):
            for q in range(NODE_NUM):
                dist[p][q] = min(dist[p][q], dist[p][r] + dist[r][q])
    result = np.sum(np.array(dist)) / (NODE_NUM * NODE_NUM - NODE_NUM)
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


def clustering_coefficient(G):
    cc = 0
    for i in range(1, NODE_NUM+1):
        if G.degree[i] >= 2:
            cc += one_node_clustering_coefficient(G, i)
    return cc/NODE_NUM

# 4.graph coreness
def graph_coreness(G):
    H = G.copy()
    for i in range(NODE_NUM):
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
    for i in range(NODE_NUM):
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



if __name__ == '__main__':
    edges = []
    read_data(edges)
    G = nx.Graph()
    creat_graph(G, edges)
    # draw_graph(G)
    # node_degree_distribution(G)
    # print(average_shortest_path_length(G))
    # print(clustering_coefficient(G))
    # print(graph_coreness(G))
    # print(node_coreness(G, 8))

