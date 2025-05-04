import networkx as nx
from cluster_by_relay_nodes import Cluster
import random


def turn_G_to_weight_graph(G: nx.Graph):
    w = nx.Graph()
    w.add_nodes_from(G.nodes())
    for edge in G.edges().data():
        w.add_edge(edge[0], edge[1], weight=len(edge[2].get("f_slot")))

    return w

def estimate_upper_bound(**kwargs):
    test = Cluster(**kwargs)
    G = turn_G_to_weight_graph(test.G)
    services = test.services
    # def random_balanced_connected_cut(G):
    #     nodes = list(G.nodes())
    #     total_nodes = len(nodes)
    #     target_size = total_nodes // 2
    #
    #     # 随机选择一个起始节点
    #     start_node = random.choice(nodes)
    #     partition1 = set()
    #     partition2 = set(nodes)
    #
    #     stack = [start_node]
    #     while stack and len(partition1) < target_size:
    #         node = stack.pop()
    #         if node not in partition1:
    #             partition1.add(node)
    #             partition2.remove(node)
    #             neighbors = list(G.neighbors(node))
    #             random.shuffle(neighbors)
    #             for neighbor in neighbors:
    #                 if neighbor not in partition1:
    #                     stack.append(neighbor)
    #
    #     # 如果 partition1 大小不足，从 partition2 中移动节点
    #     while len(partition1) < target_size:
    #         node = partition2.pop()
    #         partition1.add(node)
    #
    #     # 如果 partition1 大小超过，移回节点到 partition2
    #     while len(partition1) > target_size:
    #         node = partition1.pop()
    #         partition2.add(node)
    #
    #     cut_value = 0
    #     for u in partition1:
    #         for v in partition2:
    #             if G.has_edge(u, v):
    #                 cut_value += G[u][v].get('weight', 1)
    #     return cut_value, (list(partition1), list(partition2))
    #
    #
    # failure = 0
    #
    # def process(G1):
    #     global failure
    #     if G1.number_of_nodes() == 1 :
    #         return
    #     cut_val, partition = random_balanced_connected_cut(G1)
    #     a, b = set(partition[0]), set(partition[1])
    #     all_request = 0
    #     for service in services:
    #         if (service['snk'] in a and service['src'] in b) or (service['snk'] in b and service['src'] in a):
    #             all_request += 1
    #     if all_request > cut_val:
    #         failure += all_request - cut_val
    #     else:
    #         return
    #     process(nx.subgraph(G, partition[0]))
    #     process(nx.subgraph(G, partition[1]))
    #     return partition
    # process(G)
    # print(failure)


    total_weight = 0
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1)
        total_weight += weight

    lst = []
    for service in services:
        s = service['src']
        t = service['snk']
        shortest_path_length = nx.shortest_path_length(G, source=s, target=t)
        lst.append(shortest_path_length)

    lst.sort()
    for i in range(len(lst)):
        total_weight -= lst[i]
        if total_weight < 0:
            break

    return i





