
import networkx as nx

from cluster_by_relay_nodes import Cluster
import random
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def turn_G_to_weight_graph(G: nx.Graph):
    w = nx.Graph()
    w.add_nodes_from(G.nodes())
    for edge in G.edges().data():
        w.add_edge(edge[0], edge[1], weight=len(edge[2].get("f_slot")))

    return w

def graph_cut(G):
    """
    计算无向带权图的割，并返回分割方法和割的值。

    参数:
        G (nx.Graph): 输入的无向带权图

    返回:
        partition (tuple): 包含两个子元组，每个子元组是一个集合中的节点
        cut_val (float): 割的值，即两个集合之间边的权重之和
    """
    # 获取图的邻接矩阵（带权）
    A = nx.adjacency_matrix(G)
    A = A.tocsr()

    # 获取图的度矩阵（带权）
    D = np.diag([sum(data['weight'] for _, _, data in G.edges(node, data=True)) for node in G.nodes])
    D = csr_matrix(D)

    # 计算归一化拉普拉斯矩阵 L = I - D^(-1/2) * A * D^(-1/2)
    D_inv_sqrt = np.diag([1.0 / np.sqrt(D[node, node]) if D[node, node] > 0 else 0 for node in range(len(G.nodes))])
    D_inv_sqrt = csr_matrix(D_inv_sqrt)
    L = csr_matrix(np.eye(len(G.nodes))) - D_inv_sqrt @ A @ D_inv_sqrt

    # 求解 L 的特征值和特征向量
    eigenvalues, eigenvectors = eigsh(L, k=2, which='SM')

    # 选择第二小的特征向量（第一个特征向量通常是全1向量）
    second_smallest_eigenvector = eigenvectors[:, 1]

    # 根据特征向量的值对节点进行分割
    threshold = np.median(second_smallest_eigenvector)
    partition_0 = [node for node in G.nodes if second_smallest_eigenvector[list(G.nodes).index(node)] < threshold]
    partition_1 = [node for node in G.nodes if second_smallest_eigenvector[list(G.nodes).index(node)] >= threshold]

    # 将分割结果转换为元组形式
    partition = (tuple(partition_0), tuple(partition_1))

    # 计算割的值（带权）
    cut_val = 0.0
    for (u, v, data) in G.edges(data=True):
        if (u in partition_0 and v in partition_1) or (u in partition_1 and v in partition_0):
            cut_val += data.get('weight', 1.0)  # 使用边的权重，如果没有权重则默认为1

    return partition, cut_val



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

    bound2 = estimate_bound(**kwargs)
    return min(i, bound2)

failure = 0
def estimate_bound(**kwargs):
    test = Cluster(**kwargs)
    G = turn_G_to_weight_graph(test.G)
    services = test.services
    global failure
    def process(G1):
        global failure
        if G1.number_of_nodes() == 1 :
            return
        partition, cut_val = graph_cut(G1)
        a, b = set(partition[0]), set(partition[1])
        all_request = 0
        for service in services:
            if (service['snk'] in a and service['src'] in b) or (service['snk'] in b and service['src'] in a):
                all_request += 1
        if all_request > cut_val:
            failure += all_request - cut_val
        else:
            return
        process(nx.subgraph(G, partition[0]))
        process(nx.subgraph(G, partition[1]))
        return partition
    process(G)
    return len(services) - failure

if __name__ == '__main__':
    import service_recovery_3_25

    args = {"distance_margin": 1000,
            "ots_margin": 10,
            "osnr_margin": 0.01,
            "file_path": "new_example/",
            "file_name": "example_200",
            "band": 8,
            "c_max": 964,

            "FILE_ADD": ""
            }
    args.update({"subgraph_file_path": './subgraphs1/' + args["file_name"] + '/'})
    s = service_recovery_3_25.ServiceRecovery(**args)
    s.G = turn_G_to_weight_graph(s.G)
    estimate_bound(**args)


    print(estimate_upper_bound(**args))
    print(len(s.services) - failure)



