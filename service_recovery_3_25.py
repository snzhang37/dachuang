import math
import random
from copy import deepcopy

import networkx as nx
import numpy as np
from cluster4 import *
import itertools
import heapq
import pandas as pd
from itertools import islice
from itertools import groupby
from networkx import NetworkXNoPath
from networkx import NetworkXError
from networkx import NodeNotFound
from functools import reduce

from cluster_by_relay_nodes import Cluster
# import networkit as nk
import service_recovery_version2


import multiprocessing


# 算法流程：
# 首先，程序会通过域间路由找到子图之间的可用路径（域间路由）。
# 然后在每个子图内部进行频隙资源的分配（域内路由）。
# 最终，如果找到了合适的路径并且所有频隙资源可用，便更新网络状态，表示该服务请求已成功部署。

# 3.25更新
# 修改点 :
# 1. 将部分超参数改为类属性，方便以后实验调试
# 2. 尝试在事先解出路径和频谱

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)


seed = 1
set_seed(seed)

class ServiceRecovery(Cluster):
    def __init__(self, distance_margin, ots_margin, osnr_margin, subgraph_file_path, file_path='example/',
                 file_name='example3', band=24, c_max=964, FILE_ADD='RSOP_50/'):

        super().__init__(file_name, file_path, distance_margin, ots_margin, osnr_margin, band, c_max)
        self.distance_margin = distance_margin
        self.ots_margin = ots_margin
        self.osnr_margin = osnr_margin

        self.file_name = file_name
        self.file_path = file_path

        # self.relay_nodes = None
        # self.nodes_reachable = None
        # self.nodes_relay_reachable = None
        # self.subgraph = None
        # self.subgraph_pairs_length = None
        # self.subgraph_path_dict = None
        # self.subgraph_edge_path_dict = None
        # self.G = None
        #
        # save_name = "set_" + file_name + ".pkl"
        # Cluster.load_set_from_pkl(save_name, self)

        self.services = process_service(file_path + FILE_ADD, file_name, band, c_max) # 业务信息
        # self.relay_count, self.relay_index = self.get_relay_node() # 原图中继节点信息（数量+索引）

        self.N = self.G.number_of_nodes()  # 节点数量
        self.M = self.G.number_of_edges()  # 边数量
        self.L = self.G.graph['L']

        self.num_of_path = 20


    def update_spectrum(self, path, l):
        for i in range(len(path) - 1):
            self.G[path[i]][path[i + 1]]['f_slot'].remove(l)

    def recover_spectrum(self, path, l):
        for i in range(len(path) - 1):
            self.G[path[i]][path[i + 1]]['f_slot'].add(l)

    def sorted_services(self, services):
        length_list = []
        for service in services:
            try:
                path_length = nx.shortest_path_length(self.G, service['src'], service['snk'])
            except nx.NetworkXNoPath:
                path_length = float('inf')
            length_list.append(path_length)

        combined_list = list(zip(length_list, services))
        sorted_combined_list = sorted(combined_list, key=lambda x: x[0])
        result = [service for _, service in sorted_combined_list]
        return result

    def sorted_services_2(self, services):
        num_of_relay = []
        for service in services:
            s = service['src']
            t = service['snk']
            if s in self.nodes_reachable[t]:
                num_of_relay.append(0)
                continue

            s_domains = self.nodes_relay_reachable[s]
            t_domains = self.nodes_relay_reachable[t]
            all_pairs = list(itertools.product(s_domains, t_domains))
            if len(all_pairs) == 0:
                num_of_relay.append(float('inf'))
            else:
                num_of_relay.append(min(map(lambda x : self.subgraph_pairs_length[x[0]][x[1]], all_pairs)))

        combined_list = list(zip(num_of_relay, services))
        sorted_combined_list = sorted(combined_list, key=lambda x: x[0])
        result = [service for _, service in sorted_combined_list]
        return result

    def sorted_services_3(self, services):
        num_of_relay = []
        length_list = []
        for service in services:
            s = service['src']
            t = service['snk']
            if s in self.nodes_reachable[t]:
                num_of_relay.append(0)
            else:

                s_domains = self.nodes_relay_reachable[s]
                t_domains = self.nodes_relay_reachable[t]
                all_pairs = list(itertools.product(s_domains, t_domains))
                if len(all_pairs) == 0:
                    num_of_relay.append(float('inf'))
                else:
                    num_of_relay.append(min(map(lambda x : self.subgraph_pairs_length[x[0]][x[1]], all_pairs)))
            try:
                path_length = nx.shortest_path_length(self.G, s, t)
            except nx.NetworkXNoPath:
                path_length = float('inf')
            length_list.append(path_length)

        combined_list = list(zip(length_list, num_of_relay, services))
        sorted_combined_list = sorted(combined_list, key=lambda x: (x[1],x[0]))
        result = [service for _, _, service in sorted_combined_list]
        return result

    # 检查业务路由的正确性，从路径有效/中继器有效/频谱有效/光参有效四个方面来判断
    def get_resource_occupation(self):
        free_slot = 0
        for u, v in self.G.edges():
            free_slot += len(self.G[u][v]['f_slot'])

        return 1 - free_slot / (self.L * len(self.G.edges()))

    def find_path(self, u, v):
        if (u, v) in self.subgraph_edge_path_dict:
            # if len(self.subgraph_edge_path_dict[(u, v)][0]) > 0:
            return self.subgraph_edge_path_dict[(u, v)]
        if (v, u) in self.subgraph_edge_path_dict:
            return self.subgraph_edge_path_dict[(v, u)]
            # return [path[::-1] for path in self.subgraph_edge_path_dict[(v,u)][0]], self.subgraph_edge_path_dict[(v, u)][1]

        paths = list(islice(nx.shortest_simple_paths(self.G, u, v), self.num_of_path))

        all_paths = []
        all_spectrum = []
        for path in paths:
            if self.verify(path):
                available_l = self.alloc_spectrum_for_path(path)
                if len(available_l) > 0:
                    all_paths.append(path)
                    all_spectrum.append(available_l)

        if len(all_paths) == 0:
            if u in self.nodes_relay_reachable[v]:
                self.nodes_relay_reachable[v].discard(u)
            if v in self.nodes_relay_reachable[u]:
                self.nodes_relay_reachable[u].discard(v)
            if (u, v) in self.subgraph_pairs_length:
                self.subgraph_pairs_length[(u, v)] = float('inf')
                self.subgraph_pairs_length[(v, u)] = float('inf')
            return [], []

        if (u, v) not in self.subgraph_edge_path_dict and (v, u) not in self.subgraph_edge_path_dict:
            self.subgraph_edge_path_dict[(u, v)] = (all_paths, all_spectrum)
        return all_paths, all_spectrum

    def is_spectrum_available_for_path(self, path, l):
        for i in range(len(path) - 1):
            if l not in self.G[path[i]][path[i + 1]]['f_slot']:
                return False

        return True

    def choose_path(self, final_path):
        """choose the spectrum and update the graph and subgraph \n
        return : route if the spectrum is allocated successfully, None otherwise"""

        route = []
        found = False
        useless_l = []

        for paths, available_ls in final_path:
            found = False
            for i in range(len(paths)):
                if found:
                    break
                path = paths[i]
                available_l = available_ls[i]

                for l in available_l:
                    if self.is_spectrum_available_for_path(path, l):
                        self.update_spectrum(path, l)
                        route.append((i, path, l))
                        found = True
                        break
                    else:
                        useless_l.append((i, path[0], path[-1], l))

            if not found:
                useful_l = set()
                for i, p2, l2 in route:
                    self.recover_spectrum(p2, l2)
                    useful_l.add(l2)

                for i, u, v, l in reversed(useless_l):
                    if (u, v) in self.subgraph_edge_path_dict and l not in useful_l:
                        self.subgraph_edge_path_dict[(u, v)][1][i].discard(l)
                        if len(self.subgraph_edge_path_dict[(u, v)][1][i]) == 0:
                            self.subgraph_edge_path_dict[(u, v)][1].pop(i)
                            self.subgraph_edge_path_dict[(u, v)][0].pop(i)

                    if (v, u) in self.subgraph_edge_path_dict and l not in useful_l:
                        self.subgraph_edge_path_dict[(v, u)][1][i].discard(l)
                        if len(self.subgraph_edge_path_dict[(v, u)][1][i]) == 0:
                            self.subgraph_edge_path_dict[(v, u)][1].pop(i)
                            self.subgraph_edge_path_dict[(v, u)][0].pop(i)
                return None

        for i, p, l in reversed(route):
            if (p[0], p[-1]) in self.subgraph_edge_path_dict:
                self.subgraph_edge_path_dict[(p[0], p[-1])][1][i].discard(l)
                if len(self.subgraph_edge_path_dict[(p[0], p[-1])][1][i]) == 0:
                    self.subgraph_edge_path_dict[(p[0], p[-1])][1].pop(i)
                    self.subgraph_edge_path_dict[(p[0], p[-1])][0].pop(i)

            if (p[-1], p[0]) in self.subgraph_edge_path_dict:
                self.subgraph_edge_path_dict[(p[-1], p[0])][1][i].discard(l)
                if len(self.subgraph_edge_path_dict[(p[-1], p[0])][1][i]) == 0:
                    self.subgraph_edge_path_dict[(p[-1], p[0])][1].pop(i)
                    self.subgraph_edge_path_dict[(p[-1], p[0])][0].pop(i)

        for i, u, v, l in reversed(useless_l):
            if (u, v) in self.subgraph_edge_path_dict:
                self.subgraph_edge_path_dict[(u, v)][1][i].discard(l)
                if len(self.subgraph_edge_path_dict[(u, v)][1][i]) == 0:
                    self.subgraph_edge_path_dict[(u, v)][1].pop(i)
                    self.subgraph_edge_path_dict[(u, v)][0].pop(i)

            if (v, u) in self.subgraph_edge_path_dict:
                self.subgraph_edge_path_dict[(v, u)][1][i].discard(l)
                if len(self.subgraph_edge_path_dict[(v, u)][1][i]) == 0:
                    self.subgraph_edge_path_dict[(v, u)][1].pop(i)
                    self.subgraph_edge_path_dict[(v, u)][0].pop(i)


        return route

    def run(self, method=1):

        num_succeed = 0
        # print(self.get_resource_occupation())

        START = time.time()
        if method == 1:
            self.services = self.sorted_services(self.services)
        elif method == 2:
            self.services = self.sorted_services_2(self.services)
        elif method == 3:
            self.services = self.sorted_services_3(self.services)
        else:
            random.shuffle(self.services)
        for index, service in enumerate(self.services):

            #if num_succeed % 100 == 0:
                #print(num_succeed, "NUMOFSUCCEED")
            # print(index)
            u = service['src']
            v = service['snk']
            if u in self.nodes_reachable[v]:
                paths = self.find_path(u, v)
                if paths[0] is not None and len(paths[0]) > 0:
                    if self.choose_path([paths]) is not None:
                        num_succeed += 1
                        #print("success")
                        continue
                else:
                    self.nodes_reachable[v].discard(u)
                    self.nodes_reachable[u].discard(v)

            u_domains = self.nodes_relay_reachable[u]
            v_domains = self.nodes_relay_reachable[v]
            all_pairs = list(itertools.product(u_domains, v_domains))

            all_pairs.sort(key=lambda x: self.subgraph_pairs_length[x[0]][x[1]])
            service_end = False

            for i, j in all_pairs[:10]:
                if service_end:
                    break

                domain_routes = self.subgraph_path_dict[(i,j)][:20]
                for domain_route in domain_routes:
                    final_path = []
                    start_path = self.find_path(u, i)
                    if len(start_path[0]) == 0:
                        continue
                    final_path.append(start_path)

                    for k in range(len(domain_route) - 1):
                        # mid_paths = self.subgraph_edge_path_dict[(domain_route[k], domain_route[k + 1])]
                        mid_paths = self.find_path(domain_route[k], domain_route[k + 1])
                        if len(mid_paths[0]) == 0 :
                            break
                        final_path.append(mid_paths)

                    else:
                        end_path = self.find_path(j, v)
                        if len(end_path[0]) == 0:
                            continue
                        final_path.append(end_path)

                        if self.choose_path(final_path) is not None:
                            num_succeed += 1
                            service_end = True
                           # print("success")
                            break




        END = time.time()
        # print(f"平均业务处理时间 = {(END - START)/len(self.services)}")
        # # print(f"域级拆分更新次数 = {update_times}")
        # # print(f"平均域级更新时间 = {np.mean(time_for_update)}")
        # # resource_occupation_after = self.resource_occupation(self.G)
        # print('num_succeed:', num_succeed)
        # print(self.get_resource_occupation())
        # print(f"Service success rate: {num_succeed / len(self.services)}%")
        return num_succeed, len(self.services), (END - START)/len(self.services), self.get_resource_occupation()
        # print('ave time (success):', np.mean(time_succeed_list))
        # print('time indomain:', np.mean(time_indomain_list))
        # print('len domain route:', np.mean(len_domain_list))
        # print('time for service:', np.mean(time_for_service))
        # print(f"resource occupation before: {resource_occupation_before}, resource occupation after: {resource_occupation_after}")
        # print(len_domain_list)
        # print(success_rate_list)
        # print(domain_portion_list)
        # print(indomain_portion_list)
        # print(f"最短路径平均长度 = {np.mean(shortest_path_len)}")
        # print(f"域间处理最长时间 = {np.max(time_domain_list)}, 最短时间 = {np.min(time_domain_list)}，平均时间 = {np.mean(time_domain_list)}")
        # print(f"域内处理最长时间 = {np.max(time_indomain_list)}, 最短时间 = {np.min(time_indomain_list)}，平均时间 = {np.mean(time_indomain_list)}")




if __name__ == '__main__':
    FILE_ADD = 'RSOP_20/'

    # # for example1
    # file_name='example1'
    # file_path='example/'
    # band = 24
    # c_max = 964
    # subgraph_file_path = './subgraphs1/'+ FILE_ADD +'example1_1/'


    # # for example2
    # file_name = 'example2'
    # file_path = 'example/'
    # band = 8
    # c_max = 864
    # subgraph_file_path = './subgraphs1/'+ FILE_ADD +'example2_2/'

    # 均匀/不均匀小规模 6/3



    # # 均匀/不均匀大规模 5/4
    # file_name='example7'
    # file_path='example/'
    # band = 8
    # c_max = 968
    # subgraph_file_path = './subgraphs1/' + FILE_ADD + 'example7_7/'


    # # 均匀/不均匀中规模 7/8
    # file_name='example3'
    # file_path='example/'
    # band = 8
    # c_max = 968
    # subgraph_file_path = './subgraphs1/' + FILE_ADD + 'example3_3/'
    file_name = 'example3'
    file_path = 'example/'
    band = 8
    c_max = 968
    subgraph_file_path = './subgraphs1/' + FILE_ADD + 'example3_3/'

    distance_margin = 10 *100
    ots_margin = 10 * 1
    osnr_margin = 10 * 0.001

    S = ServiceRecovery(distance_margin, ots_margin, osnr_margin, subgraph_file_path=subgraph_file_path, file_path=file_path, file_name=file_name, band=band, c_max = c_max, FILE_ADD = FILE_ADD)
    S.run()

    # end_time = time.time()