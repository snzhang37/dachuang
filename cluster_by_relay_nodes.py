from itertools import islice

import networkx as nx
from collections import deque
from copy import deepcopy
from networkx.algorithms.simple_paths import all_simple_paths
from networkx.classes import subgraph

from Date_processing import *
import matplotlib.pyplot as plt
import time
import os
import itertools
from networkx.readwrite import json_graph
import json
import pickle

class Cluster:

    def __init__(self, file_name , file_path, distance_margin, ots_margin, osnr_margin, band, c_max, path='./subgraphs/',**kwargs):
        self.file_name = file_name
        self.distance_margin = distance_margin
        self.ots_margin = ots_margin
        self.osnr_margin = osnr_margin
        self.file_path = file_path
        self.G = create_topology(file_path=self.file_path, file_name=file_name, band=band, c_max=c_max)
        self.__modify_G()
        self.G = nx.Graph(self.G)
        self.path = path
        self.save_file_name = "set_" + self.file_name + ".pkl"


        self.services = process_service(self.file_path, self.file_name)

        self.N = self.G.number_of_nodes()  # 节点数量
        self.M = self.G.number_of_edges()  # 边数量
        self.L = self.G.graph['L']


        if os.path.exists(self.save_file_name):
            self.load_set_from_pkl(self.save_file_name, self)
        else:
            self.relay_nodes = set(get_relay_node(self.G))
            '''a set that contains all relay nodes. Set is used here because it is convenient to do intersection'''
            self.nodes_relay_reachable = {key : set() for key in self.G.nodes()}
            '''a dict that key is node and the value is a set which contain the relay nodes it can reach'''
            self.nodes_reachable = {key : set() for key in self.G.nodes()}
            self.subgraph = nx.Graph()
            self.__get_reachable()
            print(1)
            self.subgraph_pairs_length = dict(nx.shortest_path_length(self.subgraph))
            self.subgraph_path_dict = {}
            self.subgraph_edge_path_dict = {}
            self.get_subgraph_path()
            '''a subgraph that only contains the relay nodes'''
            self.save_set_to_pkl()


    def subgraph_distance(self, u, v):
        '''返回u,v在self.subgraph中的最短路径\n
        如果u,v在subgraph中不连通，返回2*N作为结果'''
        return self.subgraph_pairs_length.get(u,{}).get(v, 2*self.N)


    def alloc_spectrum_for_path(self, path) -> set[int]:
        available_l = set(i for i in range(self.L))
        for i in range(len(path) - 1):
            available_l = available_l & self.G[path[i]][path[i + 1]]['f_slot']

        return available_l


    def __search_path_in_G(self, s, t):
        '''只找无需中继的路径
        如果s,t之间不能直达，返回empty list
        如果s,t可以直达，返回s,t之间的所有路径, 且保证路径在osnr, margin, distance margin下合法,
        且将path按照路径长度升序排序
        '''
        if s not in self.nodes_reachable[t]:
            return []  # 表示两者之间不存在path

        # 它们之间的可行路径只会经过s.reachable和t.reachable的交集
        def find_all_paths(s, t):
            '''Finds all paths between s and t without relay.
            If s == t, it will return an empty list as there are no paths needed.'''
            if s == t:
                return []

            all_paths = []

            def dfs(current, target, path, path_weight):
                if current == target:
                    if self.verify(path):
                        all_paths.append(path[:])
                    return

                for neighbor in self.G.neighbors(current):
                    if neighbor not in path and neighbor in self.nodes_reachable[s]\
                        and neighbor in self.nodes_reachable[t]:
                        max_weight = 0
                        for edge in self.G[current][neighbor].values():
                            max_weight = max(max_weight, edge['ots'])

                        if path_weight + max_weight > self.ots_margin:
                            continue

                        path.append(neighbor)
                        dfs(neighbor, target, path, path_weight + max_weight)
                        path.pop()

            dfs(s, t, [s], 0)
            return all_paths

        paths = find_all_paths(s, t)
        paths.sort(key=len, reverse=False)
        if len(paths) == 0:
            # update the dict
            self.nodes_relay_reachable[s].discard(t)
            self.nodes_relay_reachable[t].discard(s)
            if s in self.relay_nodes and t in self.relay_nodes and \
                    self.subgraph.has_edge(s, t):
                self.subgraph.remove_edge(s, t)

        return paths

    def __modify_G(self):
        edges_to_remove = [
            (u, v, key)
            for u, v, key, data in self.G.edges(data=True, keys=True)
            if data.get('distance', 0) > self.distance_margin or
               data.get('ots', 0) > self.ots_margin or
               data.get('osnr', 0) > self.osnr_margin
        ]
        for u, v, key in edges_to_remove:
            self.G.remove_edge(u, v, key)

    def __get_reachable(self):
        self.subgraph.add_nodes_from(self.relay_nodes)
        for node_1 in self.G.nodes():
            for node_2 in self.G.nodes():
                if node_1 == node_2:
                    continue
                if self.is_reachable(node_1, node_2):
                    self.nodes_reachable[node_1].add(node_2)
                    self.nodes_reachable[node_2].add(node_1)
                    if self.G.nodes[node_1]['relay']:
                        self.nodes_relay_reachable[node_2].add(node_1)
                    if self.G.nodes[node_2]['relay']:
                        self.nodes_relay_reachable[node_1].add(node_2)
                    if self.G.nodes[node_2]['relay'] and self.G.nodes[node_1]['relay']:
                        self.subgraph.add_edge(node_1, node_2)

    def get_subgraph_path(self):
        for u in self.G.nodes():
            for v in self.nodes_reachable[u]:
                if (v, u) not in self.subgraph_edge_path_dict:
                    available_paths = []
                    available_ls = []
                    count = 0
                    for path in nx.shortest_simple_paths(self.G, source=u, target=v):
                        if count > 100:
                            break
                        if self.verify(path):
                            available_l = self.alloc_spectrum_for_path(path)

                            if len(available_l) > 0:
                                available_paths.append(path)
                                available_ls.append(available_l)
                        else:
                            count += 1

                    self.subgraph_edge_path_dict[(u, v)] = (available_paths, available_ls)
                    # self.subgraph_edge_path_dict[(v, u)] = ([path[::-1] for path in available_paths], deepcopy(available_ls[:]))

        for u in self.subgraph.nodes():
            for v in self.subgraph.nodes():
                if u == v:
                    self.subgraph_path_dict[(u,v)] = [[u]]
                    continue

                self.subgraph_path_dict[(u,v)] = list(islice(nx.shortest_simple_paths(self.subgraph, source=u, target=v), 20))

    def verify(self, path):
        total_distance = 0
        total_osnr = 0
        total_ots = 0
        if len(path) == 0:
            return True

        # 遍历路径上的每一条边
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]

            edge = self.G[path[i]][path[i + 1]]
            total_distance += edge['distance']
            total_osnr += edge['osnr']
            total_ots += edge['ots']

            # 检查当前累加的权重是否超过了给定的 margin
            if total_distance > self.distance_margin or total_osnr > self.osnr_margin or total_ots > self.ots_margin:
                return False

        return True

    def is_reachable(self, s, t):
        '''Checks if s and t can reach each other without relay.'''
        if s == t:
            return True

        stack = [(s, [s], 0)]
        while stack:
            current, path, path_weight = stack.pop()
            if current == t:
                if self.verify(path):
                    return True
            for neighbor in self.G.neighbors(current):
                if neighbor not in path:
                    max_weight = self.G[current][neighbor]['ots']
                    if path_weight + max_weight > self.ots_margin:
                        continue
                    new_path = path + [neighbor]
                    stack.append((neighbor, new_path, path_weight + max_weight))
        return False



    def save_set_to_pkl(self):
        with open(self.save_file_name, 'wb') as file:
            dicts = {"relay_nodes" : self.relay_nodes,
                     "nodes_reachable" : self.nodes_reachable,
                     "nodes_relay_reachable" : self.nodes_relay_reachable,
                     "subgraph" : self.subgraph,
                     "subgraph_pairs_length" : self.subgraph_pairs_length,
                     "subgraph_path_dict" : self.subgraph_path_dict,
                     "G" : self.G,
                     "subgraph_edge_path_dict" : self.subgraph_edge_path_dict
                     }
            pickle.dump(dicts, file)

    @staticmethod
    def load_set_from_pkl(file_name, obj):
        if os.path.exists(file_name):
            with open(file_name, 'rb') as file:
                dicts = pickle.load(file)
                obj.relay_nodes = dicts.get("relay_nodes")
                obj.nodes_reachable = dicts.get("nodes_reachable")
                obj.nodes_relay_reachable = dicts.get("nodes_relay_reachable")
                obj.subgraph = dicts.get("subgraph")
                obj.subgraph_pairs_length = dicts.get("subgraph_pairs_length")
                obj.subgraph_path_dict = dicts.get("subgraph_path_dict")
                obj.G = dicts.get("G")
                obj.subgraph_edge_path_dict = dicts.get("subgraph_edge_path_dict")

if __name__ == '__main__':
    distance_margin = 10 * 100
    ots_margin = 10 * 1
    osnr_margin = 10 * 0.001

    test = Cluster('example3',None, distance_margin, ots_margin, osnr_margin, 8, 964)
    # test.RSA()







