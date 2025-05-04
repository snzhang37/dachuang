import copy
import random

import networkx as nx
from collections import deque
from Date_processing import *
import matplotlib.pyplot as plt
import time
import os
from networkx.readwrite import json_graph
import json
import pickle
import pandas as pd
import itertools

path = './subgraphs/'
file_path ='example/'

# 声明一个全局集合
visited_set = set()

def add_element(element):
    global visited_set
    visited_set.add(element)

edge_cluster = set() # 存储当前所有子图中的所有的边的情况，保证重点不重边

def add_margin_element(n1, n2):
    global edge_cluster
    pair = tuple(sorted([n1, n2])) # 保证小的索引在前
    edge_cluster.add(pair)

def check_margin(n1, n2): # 检查这条边是否已经出现在其它类中
    global edge_cluster
    pair = tuple(sorted([n1, n2]))
    if pair in edge_cluster:
        return True
    else:
        return False


# 基于距离/OTS跳数/光信噪比进行聚类
# 聚类标准是各个光参margin的1/2，因为域间传输不一定要靠电中继，所以这样方便判断域间路径
def cluster(G, distance_margin, ots_margin , osnr_margin, ex, FILE_ADD): # ex用于指示是对第几个例子中的拓扑进行聚类
    # nx.draw(G,with_labels=True)
    # plt.show()
    subgraphs = [] # 聚类得到的子图
    num = 0  # break

    max_size = 10 # 每个聚类最大尺寸


    # 从邻居节点多的非中继节点开始聚类，有助于减少分类个数，和域间路径长度
    # 获取每个节点的度数，并按从大到小排序
    # 这个节点的度需要实时刷新的，不然产生的子图规模大小不等
    all_nodes = [node for node in G.nodes() if G.nodes[node]['relay'] == False]  # 目前还没有进行聚类的非中继节点
    all_nodes_degree2 = {node:G.degree(node) for node in all_nodes} # 获取节点的初始度
    all_nodes_degree = copy.deepcopy(all_nodes_degree2) # 防止耦合


    while len(all_nodes) != 0:
        start_node = max(all_nodes, key=lambda node: all_nodes_degree[node])  # 找到度最大的节点

        new_subgraph = nx.Graph() # 无向单重边图
        if G.nodes[start_node]['relay'] == False:# 只对非中继节点进行聚类

            if start_node in visited_set: # 一个非中继类中节点只能在一个类中
                continue

            new_subgraph = nx.Graph()
            new_subgraph.add_node(start_node)
            visited = set()
            queue = deque()
            queue.append(start_node)
            visited.add(start_node)

            while queue and len(list(new_subgraph.nodes)) <= max_size:
                current_node = queue.popleft()

                for n2 in G.neighbors(current_node): # 逐个探索外围节点

                    if check_margin(current_node, n2): # 说明这个节点不满足所有的邻居节点都在类中，属于边缘节点
                        continue

                    if G.nodes[n2]['relay'] == False:
                        # 非中继-非中继'
                        # 子图拓展
                        new_subgraph, visited, End2 = sub_extend(G, visited, new_subgraph,
                                                                        current_node, n2, distance_margin,
                                                                        ots_margin, osnr_margin)
                        # End2 == True 表示这个节点不能被聚类，则current_node是边缘节点
                        if not End2: # 类中非中继节点
                            queue.append(n2) # 继续探索其邻居节点

                            # 这对边不可能出现在其他域中，所以两个端节点的度都需要减一
                            all_nodes_degree[current_node] -= 1
                            all_nodes_degree[n2] -= 1

                            G.nodes[n2]['degree'] += 1
                            G.nodes[current_node]['degree'] += 1

                            add_margin_element(current_node, n2)


                    else:
                        # 非中继-中继
                        new_subgraph, visited, End2 = sub_extend(G, visited, new_subgraph,
                                                                 current_node, n2, distance_margin,
                                                                 ots_margin, osnr_margin)
                        if not End2: # 类中中继节点
                            # 这对边不可能出现在其他域中，所以两个端节点的度都需要减一
                            all_nodes_degree[current_node] -= 1
                            # all_nodes_degree[n2] -= 1 # n2是中继节点，本身就不再all_nodes中

                            G.nodes[n2]['degree'] += 1
                            G.nodes[current_node]['degree'] += 1

                            add_margin_element(current_node, n2)


                if G.degree(current_node) == new_subgraph.degree(current_node): # 即current_node所有邻居节点都在这个类中，则下次不再考虑current_node，因为一个非边缘非中继节点只能属于一个类
                    if current_node in all_nodes:
                        all_nodes.remove(current_node)



        if all(not set(new_subgraph.edges()).intersection(set(sg.edges())) for sg in subgraphs) and not nx.is_empty(
                new_subgraph): # 每个子图不能有重复的边，但可以有相同的边缘节点
            print(num)
            num += 1

            new_subgraph.graph['is_allrelay'] = False  # 表示这个域中中继节点位于边缘

            subgraphs.append(new_subgraph)

        if start_node in all_nodes:
            all_nodes.remove(start_node)


    # 不是所有邻居节点都在类中的非中继/中继节点，每个节点形成一个小子图
    count_single = 0 # 单个节点成子图的数量
    visited_relay = []  # 记录访问过的中继节点

    while len(edge_cluster) != len(list(G.edges)):
        for i in G.nodes():
            if G.nodes[i]['degree'] < G.degree(i):
                if G.nodes[i]['relay'] == True:

                    new_subgraph = nx.Graph()
                    new_subgraph.add_node(i)

                    for n2 in G.neighbors(i):
                        if check_margin(i, n2):
                            continue

                        indexes = G.get_edge_data(i, n2)[0]

                        # 对于 i 的每个邻居节点，分别获取其中路径的最大值，合格才能加到这个字图里
                        path_distance = indexes['distance']
                        path_ots = indexes['ots']
                        path_osnr = indexes['osnr']


                        # 中继只和中继聚类
                        if G.nodes[n2]['relay'] == True:
                            if path_distance > 2 * distance_margin or path_ots > 2 * ots_margin or path_osnr > 2 * osnr_margin:
                                continue # 超标了就不算邻居节点了
                            else:
                                new_subgraph.add_node(n2)
                                edge = G[i][n2]
                                for k in range(len(edge)):
                                    new_subgraph.add_edge(i, n2, **edge[k])

                                G.nodes[n2]['degree'] += 1
                                G.nodes[i]['degree'] += 1

                                add_margin_element(i, n2)

                    if not nx.is_empty(new_subgraph):
                        new_subgraph.graph['is_allrelay'] = True # 表示这个域中全部都是中继节点

                        subgraphs.append(new_subgraph)
                        count_single += 1

                        print(count_single)
                        print(f"当前边数:{len(edge_cluster)} / {len(list(G.edges))}")

                        visited_relay.append(i)

                else: # 非中继节点
                    new_subgraph = nx.Graph()
                    new_subgraph.add_node(i)

                    for n2 in G.neighbors(i):
                        if check_margin(i, n2):
                            continue

                        indexes = G.get_edge_data(i, n2)[0]

                        # 对于 i 的每个邻居节点，分别获取其中路径的最大值，合格才能加到这个字图里
                        path_distance = indexes['distance']
                        path_ots = indexes['ots']
                        path_osnr = indexes['osnr']

                        # 非中继可以和任意节点聚类
                        # 这里聚类的标准相对宽容一些，不然好多边会被忽略
                        if path_distance > 2 * distance_margin or path_ots > 2 * ots_margin or path_osnr > 2 * osnr_margin:
                            continue # 超标了就不算邻居节点了
                        else:
                            new_subgraph.add_node(n2)
                            edge = G[i][n2]
                            for k in range(len(edge)):
                                new_subgraph.add_edge(i, n2, **edge[k])

                            G.nodes[n2]['degree'] += 1
                            G.nodes[i]['degree'] += 1

                            add_margin_element(i, n2)

                    if not nx.is_empty(new_subgraph):
                        new_subgraph.graph['is_allrelay'] = False

                        subgraphs.append(new_subgraph)
                        count_single += 1

                        print(count_single)
                        print(f"当前边数:{len(edge_cluster)} / {len(list(G.edges))}")



    print(f"单个节点成子图的个数={count_single}")

    for sub in subgraphs:
        margin_nodes = []
        for node in sub.nodes:
            if sub.degree(node) < G.degree(node):  # 节点的邻接边没有全部包含在类中，就算是边缘节点
                margin_nodes.append(node)
        sub.graph['margin_nodes'] = list(set(margin_nodes))




    # 保存子图
    if not os.path.exists('subgraphs/' + FILE_ADD + ex):
        os.makedirs('subgraphs/' + FILE_ADD + ex)
    for i in range(len(subgraphs)):
        save_pkl(path + FILE_ADD + ex + '/' + 'sg' + str(i) + '.pkl', subgraphs[i])

    return subgraphs


# 子图拓展
# def sub_extend(G, visited, queue, new_subgraph, current_node, n2, distance_margin, ots_margin, osnr_margin):
def sub_extend(G, visited, new_subgraph, current_node, n2, distance_margin, ots_margin, osnr_margin):
    End2 = False
    for d in new_subgraph.nodes():
        Skip = False
        if d == n2:
            continue
        new = new_subgraph.copy()
        edge = G[current_node][n2]
        for i in range(len(edge)):
            new.add_edge(current_node, n2, **edge[i]) # 两个节点间可能存在多条边
        if nx.has_path(new, n2, d):
            path = nx.dijkstra_path(new, n2, d)
            # 使用多重边中各项指标的最大值，这个确实有点不合适，但是为了简化算法
            path_distance, path_ots, path_osnr = 0, 0, 0
            for i in range(len(path) - 1):
                indexes = new.get_edge_data(path[i], path[i + 1])

                path_distance += indexes['distance']
                path_ots += indexes['ots']
                path_osnr += indexes['osnr']
            if path_distance > distance_margin or path_ots > ots_margin or path_osnr > osnr_margin:
                Skip = True
                break
    if not Skip: # 要进入这个聚类的节点必须与类中所有节点的关系指标都不超过界限，且不在其他类中
        # if n2 not in visited:
        visited.add(n2)
        # add_element(n2)
        # queue.append(n2)
        edge = G[current_node][n2]
        for i in range(len(edge)):
            new_subgraph.add_edge(current_node, n2, **edge[i])

        add_margin_element(current_node, n2) # 这条边已经属于某一个类，不能出现在其他类中

    else:
        End2 = True

    # return new_subgraph, visited, queue, End2
    return new_subgraph, visited, End2


def save_pkl(filename, G):
    with open(filename, 'wb') as f:
        pickle.dump(G, f)


def read_pkl_file(filename):
    with open(filename, 'rb') as f:
        G = pickle.load(f)
    return G


def load_sg(path):
    subgraphs = []
    num_files = 0
    for root, dirs, files in os.walk(path):
        num_files += len(files)
    for i in range(num_files):
        sg = read_pkl_file(path + 'sg' + str(i) + '.pkl')
        subgraphs.append(sg)
    return subgraphs

# 获得中继节点的情况（数量，具体索引）
def get_relay(file_path, file_name):
    filename = file_path + file_name + '.relay.csv'
    df = pd.read_csv(filename)
    unique_values_count = df['nodeId'].nunique()  # 中继节点的数量
    unique_values = df['nodeId'].unique()  # 所有中继节点的索引
    return unique_values_count, unique_values



if __name__ == '__main__':
    FILE_ADD = 'RSOP_80/'

    # file_name='example1' # 0.7比较合适
    # file_name_sub = 'example1_1'
    # G=create_topology(file_path='example/',file_name=file_name,band=24,c_max = 964)

    # file_name = 'example2' # 0.8比较合适
    # file_name_sub = 'example2_2'
    # G = create_topology(file_path='example/', file_name=file_name, band=8, c_max=864)

    file_name = 'example3' # 0.6是一个普适参数 # 均匀/不均匀小规模 6/3
    file_name_sub = 'example3_3'
    G = create_topology(file_path='example/' + FILE_ADD, file_name=file_name, band=8, c_max=968)

    # file_name = 'example5' # 0.6是一个普适参数 # 均匀/不均匀大规模 5/4
    # file_name_sub = 'example5_5'
    # G = create_topology(file_path='example/' + FILE_ADD, file_name=file_name, band=24, c_max=984)

    # for example2
    # distance_margin = 30 * 154.7047511
    # ots_margin = 30 * 2.63800905
    # osnr_margin = 30 * 0.00556831

    # # EXAMPLE5的话改小一半，要不聚类也太慢了
    # distance_margin = 5 * 100
    # ots_margin = 5 * 1
    # osnr_margin = 5 * 0.001

    distance_margin = 8 *100
    ots_margin = 10 * 1
    osnr_margin = 10 * 0.001


    subgraphs = []
    start_time = time.time()

    # 聚类标准是各个光参margin的1/2，因为域间传输不一定要靠电中继，所以这样方便判断域间路径
    subgraphs = cluster(G, distance_margin * 0.7, ots_margin * 0.7, osnr_margin * 0.7, file_name_sub, FILE_ADD)

    end_time = time.time()
    subgraphs= load_sg(path+ FILE_ADD + file_name_sub +'/')
    print("Time:", end_time - start_time)
    print("end")
    # subgraphs = cluster(G, distance_margin, ots_margin, osnr_margin, file_name)
    print(subgraphs)
    edge_count = 0
    for sub in subgraphs:
        edge_count += len(list(sub.edges))
    print(f"聚类后所有子图的边数 = {edge_count}/ 原拓扑边数 = {len(list(G.edges))}")