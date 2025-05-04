import networkx as nx
from collections import deque
from Date_processing import *
import matplotlib.pyplot as plt
import time
import os
from networkx.readwrite import json_graph
import json
import pickle

path = './subgraphs/'


def cluster(G, distance_margin, ots_margin, osnr_margin, ex):
    # nx.draw(G,with_labels=True)
    # plt.show()
    subgraphs = []
    num = 0  # break

    for start_node in G.nodes():
        new_subgraph = nx.MultiGraph()
        if G.nodes[start_node]['relay'] == False:
            new_subgraph = nx.MultiGraph()
            new_subgraph.add_node(start_node)
            visited = set()
            queue = deque()
            queue.append(start_node)
            visited.add(start_node)
            while queue:
                current_node = queue.popleft()
                End = False
                for n2 in G.neighbors(current_node):
                    if not End:
                        if G.nodes[n2]['relay'] == False:
                            # 非中继-非中继'
                            End = False
                            new_subgraph, visited, queue, End2 = sub_extend(G, visited, queue, new_subgraph,
                                                                            current_node, n2, distance_margin,
                                                                            ots_margin, osnr_margin)
                            if End2:
                                break
                        else:
                            # 非中继-中继
                            End = True
                            Skip = False
                            new_subgraph, visited, queue, End2 = sub_extend(G, visited, queue, new_subgraph,
                                                                            current_node, n2, distance_margin,
                                                                            ots_margin, osnr_margin)

        if all(not set(new_subgraph.edges()).intersection(set(sg.edges())) for sg in subgraphs) and not nx.is_empty(
                new_subgraph):
            print(num)
            num += 1
            subgraphs.append(new_subgraph)
            # nx.draw(new_subgraph,with_labels=True)
            # plt.show()
            # G.remove_edges_from(new_subgraph.edges())
            # if nx.diameter(new_subgraph)>8:
            #     print("error!")

    # 未经过的节点，每个节点形成一个小子图
    nodes_in_G = set(G.nodes())
    unvisited_nodes = set()
    for subgraph in subgraphs:
        unvisited_nodes.update(subgraph.nodes())
    print(len(unvisited_nodes))
    unvisited_nodes = nodes_in_G - unvisited_nodes
    for i in unvisited_nodes:
        new_subgraph = G.subgraph(i)
        subgraphs.append(new_subgraph)

    # 记录每个子图中可作为中继节点的具体节点
    for i in range(len(subgraphs)):
        subgraphs[i].graph['relay_in_subgraphs'] = []
        subgraphs[i].graph['subgraph_id'] = i
        for n in subgraphs[i].nodes():
            if G.nodes[n]['relay'] == True:
                subgraphs[i].graph['relay_in_subgraphs'].append([n, G.nodes[n]['available relay num']])
    # 保存子图
    if not os.path.exists('subgraphs/' + ex):
        os.makedirs('subgraphs/' + ex)
    for i in range(len(subgraphs)):
        save_pkl(path + ex + '/' + 'sg' + str(i) + '.pkl', subgraphs[i])
    return subgraphs


def sub_extend(G, visited, queue, new_subgraph, current_node, n2, distance_margin, ots_margin, osnr_margin):
    End2 = False
    for d in new_subgraph.nodes():
        Skip = False
        if d == n2:
            continue
        new = new_subgraph.copy()
        edge = G[current_node][n2]
        for i in range(len(edge)):
            new.add_edge(current_node, n2, **edge[i])
        if nx.has_path(new, n2, d):
            path = nx.dijkstra_path(new, n2, d)
            # 使用多重边中各项指标的最大值
            path_distance, path_ots, path_osnr = 0, 0, 0
            for i in range(len(path) - 1):
                path_distance += max(new.get_edge_data(path[i], path[i + 1])[j]['distance'] for j in
                                     range(len(new.get_edge_data(path[i], path[i + 1]))))
                path_ots += max(new.get_edge_data(path[i], path[i + 1])[j]['ots'] for j in
                                range(len(new.get_edge_data(path[i], path[i + 1]))))
                path_osnr += max(new.get_edge_data(path[i], path[i + 1])[j]['osnr'] for j in
                                 range(len(new.get_edge_data(path[i], path[i + 1]))))
            if path_distance > distance_margin or path_ots > ots_margin or path_osnr > osnr_margin:
                Skip = True
                break
    if not Skip:
        if n2 not in visited:
            visited.add(n2)
            queue.append(n2)
            edge = G[current_node][n2]
            for i in range(len(edge)):
                new_subgraph.add_edge(current_node, n2, **edge[i])

    else:
        End2 = True

    return new_subgraph, visited, queue, End2


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


if __name__ == '__main__':
    file_name='example1'
    G=create_topology(file_path='example/',file_name=file_name,band=24,c_max = 964)
    # file_name = 'example2'
    # G = create_topology(file_path='example/', file_name=file_name, band=8,c_max=864)
    distance_margin = 800
    ots_margin = 10
    osnr_margin = 0.01
    # subgraphs = []
    # start_time = time.time()
    subgraphs = cluster(G,distance_margin,ots_margin,osnr_margin,file_name)
    # end_time = time.time()
    # subgraphs=load_sg(path+file_name+'/')
    # print("Time:", end_time - start_time)
    # print("end")
    # subgraphs = greedy_cluster(G, file_path='example/', file_name=file_name)
    print(subgraphs)