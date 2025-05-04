import math
import random

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
import networkit as nk


import multiprocessing


# 算法流程：
# 首先，程序会通过域间路由找到子图之间的可用路径（域间路由）。
# 然后在每个子图内部进行频隙资源的分配（域内路由）。
# 最终，如果找到了合适的路径并且所有频隙资源可用，便更新网络状态，表示该服务请求已成功部署。



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)


seed = 1
set_seed(seed)

class ServiceRecovery():
    def __init__(self, distance_margin, ots_margin, osnr_margin, subgraph_file_path, file_path='example/', file_name='example1', band=24,c_max = 964, FILE_ADD = 'RSOP_50/'):
        self.distance_margin = distance_margin
        self.ots_margin = ots_margin
        self.osnr_margin = osnr_margin

        self.file_name = file_name
        self.file_path = file_path

        self.G = create_topology(file_path + FILE_ADD, file_name, band, c_max)  # 拓扑，原图
        self.G = nx.Graph(self.G) # 变成单重边图
        self.subgraphs = load_sg(subgraph_file_path) # 载入聚类结果

        self.G_sub = self.G_domain() # 将域抽象为节点的子图，体现域间连接

        self.services = process_service(file_path + FILE_ADD, file_name, band, c_max) # 业务信息
        self.relay_count, self.relay_index = self.get_relay_node() # 原图中继节点信息（数量+索引）


        self.N = self.G.number_of_nodes()  # 节点数量
        self.M = self.G.number_of_edges()  # 边数量
        self.L = self.G.graph['L']

        # 处理节点在哪个域
        self.process_domain()

    @staticmethod
    # 路由计算
    def route(G, s, d):
        try:
            # 尝试获取最短路径长度
            # return nx.dijkstra_path(G, s, d)
            return nx.shortest_path(G, s, d)

        except NetworkXNoPath:
            # 如果无路径，返回 0
            return 0

    def get_relay_node(self):
        filename = self.file_path + FILE_ADD + self.file_name + '.relay.csv'
        df = pd.read_csv(filename)
        unique_values_count = df['nodeId'].nunique()  # 中继节点的数量
        unique_values = df['nodeId'].unique()  # 所有中继节点的索引
        return unique_values_count, unique_values

    def resource_occupation(self, G):
        free_slot = 0
        num_e = 0
        for u, v in G.edges():
            num_e += 1
            free_slot += sum(self.G[u][v]['f_slot'])

        if num_e == 0:
            return 1
        return 1 - free_slot / (num_e * self.L)

    # 处理节点在哪个域，域间重点不重边
    def process_domain(self):
        # 初始化每个子图都是有效的
        for subgraph in self.subgraphs:
            subgraph.graph['valid'] = True  # 初始化每个子图都是有效的

        for n in self.G.nodes():
            self.G.nodes[n]['domain'] = set()
            for i, sg in enumerate(self.subgraphs):
                if sg.has_node(n):
                    self.G.nodes[n]['domain'].update([i])


    def check_margin(self, path):
        total_distance = 0
        total_ots = 0
        total_osnr = 0

        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]
            total_distance += self.G[current_node][next_node]['distance']
            total_ots += self.G[current_node][next_node]['ots']
            total_osnr += self.G[current_node][next_node]['osnr']

        # 检查总距离、OTS跳数和最小OSNR是否在边际值内
        flag_within = total_distance <= self.distance_margin and total_ots <= self.ots_margin and total_osnr <= self.osnr_margin
        return flag_within, total_distance, total_ots, total_osnr


    # 对分域结果进一步处理，将域抽象为节点，并构建新的域间路径
    def G_domain(self):
        sub_num = list(range(len(self.subgraphs)))  # 生成子图的索引
        combinations = list(itertools.combinations(sub_num, 2))  # 子图两两组合的索引

        count_overlap = 0  # 记录域间链路的基础数量

        G_sub = nx.Graph()  # 单边无向图，构建域间图
        nodes = list(range(len(self.subgraphs)))
        G_sub.add_nodes_from(nodes)  # 每个子图抽象为一个节点

        for index, (i, j) in enumerate(combinations):
            sg1 = self.subgraphs[i]  # 子图1
            sg2 = self.subgraphs[j]  # 子图2
            margin1 = sg1.graph['margin_nodes']  # 子图1的边缘节点
            margin2 = sg2.graph['margin_nodes']  # 子图2的边缘节点
            common_margin = set(margin1).intersection(set(margin2))  # 获取相同的边缘节点
            common_margin = list(common_margin)
            common_relay = []  # 存储相同的边缘中继节点
            common_non_relay = []  # 存储相同的边缘非中继节点
            for n in common_margin:
                if self.G.nodes[n]['relay'] == True: # 不仅要是中继，还要有效
                    if self.G.nodes[n]['available relay num'] > 0:
                        common_relay.append(n)
                else:
                    common_non_relay.append(n)

            if len(common_margin) > 0:
                count_overlap += 1
                G_sub.add_edge(i, j)
                # print(f"子图{i}和子图{j}共有{len(common_margin)}个重合的边缘节点，其中中继节点有{len(common_relay)}个，非中继节点有{len(common_non_relay)}个。")
                if len(common_non_relay) > 0:
                    G_sub[i][j]['available_non_relay'] = common_non_relay  # 记录这两个域进行域间传输时，可用的所有非中继节点
                if len(common_relay) > 0:
                    G_sub[i][j]['available_relay'] = common_relay  # 记录这两个域进行域间传输时，可用的所有中继节点

        return G_sub


    # 使用BFS的方法寻找域间路径，可以显著减少路径跳数
    # def domain_route(G, start, end):
    def domain_route(self, service, num_route):
        # 对于每个业务请求：
        #   1. 初始阶段：对每对端点域尝试少量候选（例如1条）。
        #   2. 如果初始候选均无法路由成功，则进入扩展阶段：增加候选路径数（例如扩展到5条），
        #      并对扩展阶段设置时间上限（例如2ms），超时则返回空。

        num_of_path = 40 # 找到k条路径即停止查找
        count_path = 0 # 统计条数

        flag_1 = 1  # 标志源节点和目的节点在一个子图内

        src_domain = self.G.nodes[service['src']]['domain']  # 源域
        snk_domain = self.G.nodes[service['snk']]['domain']  # 目的域

        all_all_paths = []  # 存储每对“源-目的域”所有的路径，每种情况最多5条


        # src和snk在同一子图
        if src_domain & snk_domain:  # 检查两个domain是否有交集
            all_paths = [[i] for i in src_domain & snk_domain]
            all_all_paths.append(all_paths)
            count_path += 1
            # flag_1 = 0 # 不应中断寻找其他域间路由

        # src和snk在不同子图
        if flag_1:
            for src in src_domain:
                if count_path == num_of_path:
                    break
                for snk in snk_domain:
                    # if nx.has_path(self.G_sub, src, snk):  # 确保存在路径再开始检索，避免浪费时间进行DFS/BFS搜索

                    if self.G_sub.has_edge(src, snk):
                        all_paths = [src, snk]
                        all_all_paths.append([all_paths])

                        count_path += 1
                        if count_path == num_of_path:
                            break

                    try:  # 用内置BFS函数，速度快，但成功率会低，不知道为啥
                        # all_paths = nx.shortest_path(self.G_sub, src, snk)
                        # all_paths = list(islice(nx.shortest_simple_paths(self.G_sub, src, snk), num_route)) # 这个参数对结果也有很大的影响
                        all_paths = list(islice(nx.all_simple_paths(self.G_sub, src, snk), num_route))  # 这个参数对结果也有很大的影响

                    except NetworkXNoPath:
                        all_paths = []
                    except NodeNotFound:
                        all_paths = []

                    if len(all_paths) > 0: # 找到了有效路径
                        count_path += 1
                        # all_all_paths.append([all_paths])
                        all_all_paths.append(all_paths)

                        if count_path >= num_of_path:
                            break

                    # search_end = time.time()
                    # print(f"路径{all_paths}搜索时间 = {search_end - search_start}")

        return all_all_paths

    # 域内路由
    def indomain_route(self, subgraph_index, s, d, flag, sub_routes):
        return self._handle_domain_routing(subgraph_index, s, d, sub_routes)

    # 非域内路由逻辑封装
    def _handle_non_domain_routing(self, s, d, sub_routes,subgraph_index):
        available_layer = []
        available_route = []
        indexes = []

        sub_subroutes = []

        flag_within, distance, ots, osnr = self.check_margin([s, d])
        if flag_within:
            available_layer = self.G[s][d]['f_slot']
            available_route = [[s, d]]
            indexes = [[distance, ots, osnr]]

        if available_layer:
            sub_subroutes.append({
                'route': available_route,
                'layer': available_layer,
                # 'in_domain': False,
                'domain_index':subgraph_index,
                'indexes': indexes,
                'relay_index': -1
            })
            sub_routes.append(sub_subroutes)
            return 1, sub_routes
        return 0, sub_routes

    # 域内路由逻辑封装
    def _handle_domain_routing(self, subgraph_index, s, d, sub_routes, num_route = 1):
        if self.subgraphs[subgraph_index].graph['is_allrelay'] == True:
            route = self.route(self.subgraphs[subgraph_index], s, d)
            if route != 0:
                for i_p in range(len(route)-1):
                    s_current = route[i_p]
                    d_current = route[i_p+1]
                    flag_suc, sub_routes = self._handle_non_domain_routing(s_current, d_current, sub_routes,subgraph_index)
                    if not flag_suc:
                        return 0, sub_routes
                return 1, sub_routes
            else:
                return 0, sub_routes
        else:
            available_layer = []
            available_route = []
            indexes = []

            # route = self.route(self.subgraphs[subgraph_index], s, d)
            #
            # if route == 0:
            #     return 0, sub_routes

            routes = list(islice(nx.shortest_simple_paths(self.subgraphs[subgraph_index], s, d), num_route))
            if len(routes) == 0:
                return  0, sub_routes

            # route = route[0]

            sub_subroutes = [] # 存储这个域内的所有可用路径和路径的资源情况
            for route in routes:
                flag_within, distance, ots, osnr = self.check_margin(route)
                if flag_within:
                    available_l = [_ for _ in range(self.L)]  # 存储该段可以使用的频隙，不断取交集更新
                    for y in range(len(route) - 1):
                        available_l_slice = self.G[route[y]][route[y+1]]['f_slot']
                        available_l = list(set(available_l) & set(available_l_slice))
                        if len(available_l) == 0:  # 这条链路没有可以使用的频隙，报废
                            break
                    if len(available_l) == 0:
                        return 0, sub_routes
                    else:
                        available_route = [route]
                        indexes = [[distance, ots, osnr]]
                        available_layer = available_l

                if available_layer:
                    sub_subroutes.append({
                        'route': available_route,
                        'layer': available_layer,
                        # 'in_domain': True,
                        'domain_index':subgraph_index,
                        'indexes': indexes,
                        'relay_index': -1
                    })
            if len(sub_subroutes) != 0:
                sub_routes.append(sub_subroutes)
                return 1, sub_routes
            else:
                return 0, sub_routes

    def domain_edge_nodes(self, s_domain, d_domain): # 计算所有可以用于域间传输的节点
        connect_situ = self.G_sub.get_edge_data(s_domain, d_domain)  # 获取这条链路的信息
        non_relay_nodes_1 = []
        relay_nodes_1 = []
        # 非中继传输
        if 'available_non_relay' in connect_situ:
            non_relay_nodes_1 = connect_situ['available_non_relay']
        # 中继传输
        if 'available_relay' in connect_situ:
            relay_nodes_1 = connect_situ['available_relay']
        d_sub = non_relay_nodes_1 + relay_nodes_1
        return d_sub

    def updata_spectrum(self, path, l):
        for i in range(len(path) - 1):
            self.G[path[i]][path[i + 1]]['f_slot'].remove(l)

    # 更新网络状态，包括占用频隙和子图中的状态更新
    def updata_state(self, domain_route, sub_routes):
        wasted_relay = [] # 记录失效的中继节点
        for index_r, r in enumerate(sub_routes):
            # if len(list(set(r['route']))) == 1:
            #     continue
            route = r['route'][0]
            layer = r['layer']
            self.updata_spectrum(route, layer)

            # 改为中继节点上中继器数量无限
            # if r['relay_index'] != -1:
            #     self.G.nodes[r['route'][-1]]['available relay'][r['relay_index']]['available'] = False
            #     self.G.nodes[r['route'][-1]]['available relay num'] -= 1
            #
            #     # 检查中继节点还有没有可用的中继器
            #     if self.G.nodes[r['route'][-1]]['available relay num'] == 0:
            #         self.G.nodes[r['route'][-1]]['relay'] = False # 相当于变成一个普通节点了
            #         wasted_relay.append(r['route'][-1])



            for i in range(len(route) - 1):
                if len(self.G[route[i]][route[i+1]]['f_slot']) == 0: # 只有当这条链接没有可用频隙时才删除
                    self.subgraphs[r['domain_index']].remove_edge(route[i], route[i + 1])
                    self.G.remove_edge(route[i], route[i+1])


        # # 部分中继节点变为无效，处理这部分对域级图的影响
        # if len(wasted_relay) > 0:
        #     for w_r in wasted_relay:
        #         visited_com = [] # 避免重复访问
        #         domains = self.G.nodes[w_r]['domain'] # 该原中继节点的域
        #         for domain in domains:
        #             del_neighbors = [] # 要删除的连接
        #             for neighbor in self.G_sub.neighbors(domain):
        #                 com = sorted([domain, neighbor])
        #                 if com not in visited_com:
        #                     visited_com.append(com)
        #                     values = self.G_sub.get_edge_data(domain, neighbor)  # 获取这条链路的信息
        #                     if 'available_relay' in values:
        #                         relay_con = values['available_relay']
        #                         relay_int = [item for item in relay_con if isinstance(item, int)]
        #                         if w_r in relay_int:
        #                             relay_int.remove(w_r)
        #
        #                             # 仍有可能作为非中继节点继续用于域间传输
        #                             if self.subgraphs[domain].degree(w_r) > 0 and self.subgraphs[neighbor].degree(w_r) > 0:
        #                                 if 'available_non_relay' in values:
        #                                     if w_r not in values['available_non_relay']:
        #                                         values['available_non_relay'].append(w_r)
        #
        #                         relay_str = [item for item in relay_con if isinstance(item, str)]
        #                         str_result = [item for item in relay_str if str(w_r) not in item]
        #
        #                         self.G_sub[domain][neighbor]['available_relay'] = relay_int + str_result # 更新这个失效的中继节点所涉及到的连接
        #
        #                         if len(self.G_sub[domain][neighbor]['available_relay']) == 0:
        #                             # 没有中继连接，也没有非中继连接了，才取消域级链接
        #                             if 'available_non_relay' in values:
        #                                 if len(self.G_sub[domain][neighbor]['available_non_relay']) > 0:
        #                                     continue
        #                             del_neighbors.append((domain, neighbor))
        #
        #                 else:
        #                     continue
        #             self.G_sub.remove_edges_from(del_neighbors)


    def update_subgraph(self, domain_route): # 更新域级图，对于本次路由涉及到的各个域，检查每个域是否是连通图，这是作为域的基本条件，不满足的话需要将这个域重新聚类，并调整和邻域的关系
        new_subs = [] # 每个元素是一个字典，键是子图索引，键值是其拆分形成的子图
        for i in domain_route:
            if not nx.is_connected(self.subgraphs[i]): # 不是连通图
                self.subgraphs[i].graph['valid'] = False # 这个子图即将被拆分或丢弃，后续不再使用这个子图本身，为了减小G_sub的更新复杂度，这里不将其删除，只是将其屏蔽

                # 获取连通子图的节点集合,没什么实际作用，主要是方便调试和验证
                connected_components = list(nx.connected_components(self.subgraphs[i]))

                # 如果所有连通子图都是单节点子图，那么这个域可以不用更新直接丢弃了
                if all(len(s) == 1 for s in connected_components):
                    continue
                else: # 继续拆分和更新
                    connected_components = [s for s in connected_components if len(s) > 1]
                    # 根据节点集合还原出连通子图
                    new_sub = [self.subgraphs[i].subgraph(component).copy() for component in connected_components]
                    # 添加margin_nodes和is_allrelay属性
                    for new_s in new_sub:
                        if self.subgraphs[i].graph['is_allrelay'] == True:
                            new_s.graph['is_allrelay'] = True
                        else:
                            if all(self.G.nodes[node]['relay'] == True for node in new_s.nodes):
                                new_s.graph['is_allrelay'] = True
                            else:
                                new_s.graph['is_allrelay'] = False
                        # new_s.graph['margin_nodes'] = list(new_s.nodes)
                        margin_nodes = []
                        for node in new_s.nodes:
                            if new_s.degree(node) < self.G.degree(node):
                                margin_nodes.append(node)
                        new_s.graph['margin_nodes'] = margin_nodes
                        new_s.graph['valid'] = True # 拆分生成的子图才是可以继续使用的

                    new_subs.append(new_sub)


        # 扁平化列表
        new_subs = [item for sublist in new_subs for item in
                    (sublist if isinstance(sublist, list) else [sublist])]
        print(f"拆分形成了{len(new_subs)}个新域")

        if len(new_subs) > 0:
            self.subgraphs = self.subgraphs + new_subs

            domain_update_s = time.time()
            new_subs_index = list(range(len(self.subgraphs)-len(new_subs), len(self.subgraphs)))
            # 首先，将所有valid = False的域节点相关的连接删除
            for i in domain_route:
                if not self.subgraphs[i].graph['valid']:
                    associated_neighbors = list(self.G_sub.neighbors(i)) # 获取所有的邻居节点
                    edges_to_remove = list(self.G_sub.edges(i))  # 获取与节点2相关的所有边
                    self.G_sub.remove_edges_from(edges_to_remove)  # 删除这些边

            # 接着，为新生成的域更新域级链接
                    for a in associated_neighbors:
                        for b in new_subs_index:
                            margin_a = self.subgraphs[a].graph['margin_nodes']
                            margin_b = self.subgraphs[b].graph['margin_nodes']
                            common_margin = set(margin_a).intersection(set(margin_b))  # 获取相同的边缘节点
                            common_margin = list(common_margin)
                            common_relay = []  # 存储相同的边缘中继节点
                            common_non_relay = []  # 存储相同的边缘非中继节点
                            for n in common_margin:
                                if self.G.nodes[n]['relay'] == True:  # 不仅要是中继，还要有效
                                    if self.G.nodes[n]['available relay num'] > 0:
                                        common_relay.append(n)
                                else:
                                    common_non_relay.append(n)

                            if len(common_margin) > 0:
                                self.G_sub.add_edge(a, b)
                                if len(common_non_relay) > 0:
                                    self.G_sub[a][b]['available_non_relay'] = common_non_relay  # 记录这两个域进行域间传输时，可用的所有非中继节点
                                if len(common_relay) > 0:
                                    self.G_sub[a][b]['available_relay'] = common_relay  # 记录这两个域进行域间传输时，可用的所有中继节点

            domain_update_e = time.time()
            print(f"域级更新完成，花费时间 = {domain_update_e - domain_update_s}")

            domain_ensure_s = time.time()
            associated_nodes = []
            for i in domain_route:
                associated_nodes = associated_nodes + list(self.subgraphs[i].nodes)
            associated_nodes = list(set(associated_nodes))
            for n in associated_nodes:
                self.G.nodes[n]['domain'] = set()
                for i, sg in enumerate(self.subgraphs):
                    if sg.has_node(n) and sg.graph['valid'] == True:
                        self.G.nodes[n]['domain'].update([i])
            domain_ensure_e = time.time()
            print(f"定域更新完成，花费时间 = {domain_ensure_e - domain_ensure_s}")

    def chosen_route(self, sub_routes):
        slices = []
        for j in range(len(sub_routes)-1):
            if sub_routes[j]['relay_index'] != -1:
                slices.append(j+1)

        if len(slices) == 0: # 整条路径没有使用电中继，共享同一个可用频隙列表
            layers = sub_routes[-1]['layer']
            # chosen_index = 0
            chosen_index = layers.index(min(layers))

            for sr in sub_routes:
                sr['layer'] = layers[chosen_index]
                # sr['route'] = sr['route'][chosen_index]

        elif len(slices) == 1: # 只使用了一个电中继
            for j in range(len(slices)+1): # 如果slices中有m个元素，则总路径被划分为m+1段
                if j == len(slices):
                    paths = sub_routes[slices[-1]:]
                else:
                    paths = sub_routes[:slices[j]]


                # 计算每一段可用的共同空闲波长
                layers = [path['layer'] for path in paths]
                # 使用 reduce 动态求交集
                common_layer = list(reduce(lambda x, y: set(x) & set(y), layers)) # 共同空闲波长
                chosen_layer = min(common_layer)

                for path in paths:
                    this_layers = path['layer']
                    # chosen_index = this_layers.index(chosen_layer)

                    path['layer'] = chosen_layer
                    # path['route'] = path['route'][chosen_index]

        else: # > 1 ,使用了两个及以上电中继，分为多段
            for j in range(len(slices)+1): # 如果slices中有m个元素，则总路径被划分为m+1段
                if j == len(slices):
                    paths = sub_routes[slices[-1]:]
                else:
                    if j == 0:
                        paths = sub_routes[:slices[j]]
                    else:
                        paths = sub_routes[slices[j-1]:slices[j]]


                # 计算每一段可用的共同空闲波长
                layers = [path['layer'] for path in paths]
                # 使用 reduce 动态求交集
                common_layer = list(reduce(lambda x, y: set(x) & set(y), layers)) # 共同空闲波长
                chosen_layer = min(common_layer)

                for path in paths:
                    this_layers = path['layer']
                    # chosen_index = this_layers.index(chosen_layer)

                    path['layer'] = chosen_layer
                    # path['route'] = path['route'][chosen_index]
        return sub_routes


    def record_success(self, succ_num):
        with open('success_index.txt', 'a') as f:
            f.write(str(succ_num))
            f.write(' ')
            f.close()

    def check_margin1(self, path, sub_routes):
        total_distance = 0
        ots = 0
        total_osnr = 0

        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]

            for index_j, j in enumerate(sub_routes): # 找到对应的子路径索引
                if j['route'][-1][0] == current_node and j['route'][-1][-1] == next_node:
                    break

            total_distance += min([e[0] for e in sub_routes[index_j]['indexes']])
            ots += min([e[1] for e in sub_routes[index_j]['indexes']])
            total_osnr += min([e[2] for e in sub_routes[index_j]['indexes']])

        # 检查总距离、OTS跳数和最小OSNR是否在边际值内
        return (total_distance <= distance_margin and
                ots <= ots_margin and
                total_osnr <= osnr_margin
                )

    def find_relay(self, path, sub_routes):
        # end_node = path[-1]
        for i in range(len(path)-1, 0, -1):
            if self.check_margin1(path[:i+1], sub_routes) and (i==len(path)-1 or self.G.nodes[path[i]]['relay']):
                return i
        return 0

    def cut_path(self, path, sub_routes): # 这一步只是确定中继器的使用情况，不考虑频谱分配
        start_node = 0
        path_slices = []
        path_indexes = []
        while start_node != len(path)-1:
            end_node = self.find_relay(path[start_node:], sub_routes)
            if end_node == 0:
                return 0,0  # 无法满足约束
            path_slices.append(path[start_node:start_node+end_node+1])
            path_indexes.append(list(range(start_node,start_node+end_node+1)))
            start_node = start_node+end_node
        return path_slices, path_indexes

    # 定义一个函数来检查子列表中是否包含连续部分与 target 相同
    def contains_consecutive_sublist(self, sublist, target):
        for i in range(len(sublist) - len(target) + 1):  # 滑动窗口
            if sublist[i:i + len(target)] == target:  # 找到连续部分
                return True
        return False

    def sorted_services(self, services):
        # 先按照业务端点的距离处理顺序，越远越考前
        length_list = []
        for service in services:

            try:
                path = nx.shortest_path(self.G, service['src'], service['snk'])
            except NetworkXNoPath:
                path = []

            length_list.append(len(path))

        # sorted_indices = [index for index, value in sorted(enumerate(length_list), key=lambda x: x[1], reverse=False)]

        # 按数据大小排序并返回原索引
        sorted_list = sorted(enumerate(length_list), key=lambda x: x[1], reverse=True)
        # 分组相等的元素
        sorted_list.sort(key=lambda x: x[1])
        # 根据第二项分组，结果只保留第一项
        grouped = {key: [item[0] for item in group] for key, group in groupby(sorted_list, key=lambda x: x[1])}
        group_keys = sorted(list(grouped.keys()), reverse=True)

        sorted_indices = []
        # 对于距离相等的业务，根据其路径上的邻居节点平均值/最小值进一步细分，越小越考前
        for group in group_keys:
            service_same_dis = grouped[group] # 距离相等的业务集合
            length_list = []
            for service in service_same_dis:

                try:
                    path = nx.shortest_path(self.G, services[service]['src'], services[service]['snk'])
                    neighbor_along_path = min(len(list(self.G.neighbors(node))) for node in path)
                except NetworkXNoPath:
                    path = []
                    neighbor_along_path = 100

                # neighbor_along_path = 0
                # for node in path:
                #     neighbor_along_path += len(list(self.G.neighbors(node)))
                # length_list.append(neighbor_along_path/len(path)) # 一路上邻居节点平均值


                length_list.append(neighbor_along_path) # 一路上邻居节点最小值最小值（效果更好）

            sorted_indices1 = [service_same_dis[index] for index, value in sorted(enumerate(length_list), key=lambda x: x[1], reverse=False)]
            sorted_indices += sorted_indices1

        # 根据索引重新排序
        services = [services[i] for i in sorted_indices] # 按照阻塞概率排列的，越容易阻塞的越靠前
        return services

    # 检查业务路由的正确性，从路径有效/中继器有效/频谱有效/光参有效四个方面来判断
    def check_success(self, subroute, service):
        real_success_flag = True

        if len(subroute) != 1: # 大于一段子路径时，首先检查一下相邻两段子路径首尾是否一致
            for sr in range(len(subroute)-1):
                if subroute[sr]['route'][0][-1] != subroute[sr+1]['route'][0][0]:
                    real_success_flag = False

        # 路径连贯，则检查路径两端与业务请求端相同
        if subroute[0]['route'][0][0] != service['src'] or subroute[-1]['route'][0][-1] != service['snk']: # 路径两端等于业务请求端
            real_success_flag = False

        index_check = [0,0,0] # 保存使用中继器前的累积光参

        for sr in subroute:
            route = sr['route'][0]
            indexes = sr['indexes'][0]

            index_check[0] += indexes[0]
            index_check[1] += indexes[1]
            index_check[2] += indexes[2]
            if index_check[0] > self.distance_margin or index_check[1] > self.ots_margin or index_check[2] > self.osnr_margin:
                real_success_flag = False
                break

            # 如果使用了中继器，因为当前有所简化，只需要判断该节点是不是中继节点
            if sr['relay_index'] != -1:
                index_check = [0,0,0]
                if self.G.nodes[route[-1]]['relay'] == False:
                    real_success_flag = False
                    break

            for n in range(len(route)-1):
                if not self.G.has_edge(route[n], route[n+1]): # 链接存在，路径有效
                    real_success_flag = False
                    break
                # 链接存在空闲频隙，且使用的频隙有效，频谱有效
                available_l = self.G[route[n]][route[n+1]]['f_slot']
                if len(available_l) == 0 or sr['layer'] not in available_l:
                    real_success_flag = False
                    break

        return real_success_flag

    def process_domain_route(self, domain_route, service):
        success_flag = 0  # 域内传输成功标志
        start_routing = time.time()

        if len(domain_route) > 1:
            flag_del = True
            while flag_del:
                if len(domain_route) > 1:
                    if service['src'] in self.subgraphs[domain_route[0]].graph['margin_nodes'] and service['src'] in self.subgraphs[domain_route[1]].graph['margin_nodes']:
                        domain_route = domain_route[1:]
                    else:
                        flag_del = False

                else:
                    flag_del = False

        if len(domain_route) == 1:  # 不需要inter_domain
            indomain_success_flag, sub_routes = self.indomain_route(domain_route[0],
                                                                    service['src'],
                                                                    service['snk'],
                                                                    3,
                                                                    sub_routes=[])
            if indomain_success_flag:  # 业务路由成功
                success_flag = 1
                i = -1
            else:
                return 0, [], -1


        else:  # 需要inter_domain
            # 先找出域内路径，再考虑域间连接方式
            for i, subgraph_index in enumerate(domain_route):
                # 源域
                if i == 0:
                    indomain_success_flag = 0  # 域间传输成功标志
                    s = service['src']  # 当前节点（源节点）
                    s_d = domain_route[i]  # 当前域
                    d_d = domain_route[i + 1]  # 目标域
                    d_sub = self.domain_edge_nodes(s_d, d_d)  # 所有可以用于域间传输的节点，包括（中继， 非中继，’中继+中继‘）

                    for d in d_sub:
                        indomain_success_flag1, sub_routes = self.indomain_route(subgraph_index, s, d,
                                                                                 0, sub_routes=[])
                        if indomain_success_flag1:
                            indomain_success_flag = 1
                            break
                        else:
                            continue

                    if indomain_success_flag:  # find indomain route, next domain
                        continue
                    else:
                        break  # can't find route, next domain route


                # 目的域
                elif i == len(domain_route) - 1:
                    # s = sub_routes[-1]['route'][-1][-1]  # 上个域路由的最后节点
                    s = d
                    d = service['snk']  # 目标节点

                    if s == d:
                        success_flag = 1
                        break

                    else:
                        indomain_success_flag1, sub_routes = self.indomain_route(subgraph_index, s, d, 2,
                                                                                 sub_routes)
                        if indomain_success_flag1:
                            success_flag = 1
                        else:
                            return 0, [], -1

                # 中间域
                else:
                    indomain_success_flag = 0
                    # s = sub_routes[-1]['route'][-1][-1]
                    s = d
                    s_d = domain_route[i]
                    d_d = domain_route[i + 1]
                    d_sub = self.domain_edge_nodes(s_d, d_d)

                    # 备份一份，方便路由失败及时恢复，把字典转换成列表，不可变数据类型
                    # sub_routes_ori = [list(item.items()) for item in sub_routes]
                    # 备份，使用深拷贝保证完全独立
                    sub_routes_ori = copy.deepcopy(sub_routes)

                    for d in d_sub:
                        # sub_routes = [dict(item) for item in sub_routes_ori]

                        # 之后每次需要使用备份时，再拷贝一份
                        sub_routes = copy.deepcopy(sub_routes_ori)

                        indomain_success_flag1, sub_routes = self.indomain_route(subgraph_index, s, d,
                                                                                 1, sub_routes)
                        if indomain_success_flag1:
                            indomain_success_flag = 1
                            break
                        else:
                            continue

                    if indomain_success_flag:  # find indomain route, next domain
                        continue
                    else:
                        break  # can't find route, next domain route

        end_routing = time.time()
        # print(f"路由计算时间 = {end_routing - start_routing}")

        return success_flag, sub_routes, i


    def run(self):
        num_succeed = 0
        time_succeed_list = []
        time_indomain_list = []
        time_domain_list = []
        time_for_service = []  # 域间时间+最长域内时间
        len_domain_list = []
        resource_occupation_before = self.resource_occupation(self.G)
        print(resource_occupation_before)

        time_all = 0

        success_rate_list = []
        indomain_portion_list = []
        domain_portion_list = []
        # record_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 1800] # example1
        record_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 1800, 2000, 2500] # example2

        START = time.time()

        self.services = self.sorted_services(self.services[:2501]) # 处理服务顺序，距离远的放前面
        for index, service in enumerate(self.services):

            if index in record_list:
                success_rate_list.append(num_succeed/index)
                indomain_portion_list.append(np.mean(time_indomain_list))
                domain_portion_list.append(np.mean(time_domain_list))

            start_service = time.time()
            start_domain = time.time()

            # # 由于先处理绝对距离较远的业务，相当于先处理阻塞概率大的业务，所以这里是根据业务的序号来决定域间路径的个数。反比关系
            # num_routes = [3,2,2,1,1] # 域间路径计算可用数目
            # num_route = math.floor(len(num_routes) * (index + (len(self.services))/len(num_routes)) / len(self.services))
            # domain_routes = self.domain_route(service, num_routes[num_route-1])  # 查找域间路径

            domain_routes = self.domain_route(service, 2)  # 查找域间路径

            # 扁平化列表
            domain_routes_flat = [item for sublist in domain_routes for item in
                                  (sublist if isinstance(sublist, list) else [sublist])]
            domain_routes_flat = [item for item in domain_routes_flat if len(item) != 0 ]

            # 优先选择路径短的
            domain_routes_flat = sorted(domain_routes_flat, key=len, reverse=False)

            end_domain = time.time()
            time_domain = end_domain - start_domain
            time_domain_list.append(time_domain)

            # print(f"域间传输时间 = {time_domain}")
            # print(f"域间路径个数 = {len(domain_routes_flat)}")
            # print("\n")

            # route in G Layering for each domain in each layer (First Fit)
            if len(domain_routes_flat) > 0: # 找到路径，表示域间路由成功
                time_all += time_domain

                while len(domain_routes_flat) > 0:
                    domain_route = domain_routes_flat[0]
                    domain_routes_flat = domain_routes_flat[1:]

                    success_flag, sub_routes, end_index = self.process_domain_route(domain_route, service)

                    if success_flag: # 对于该条域间连接情况，所有的域内连接都可以成功找到
                        all_combo = [list(prod) for prod in itertools.product(*sub_routes)]

                        success_flag1 = 0
                        for sub_routes in all_combo:
                            # 进行路径抽象，将每一段端到端链路抽象为一个直接链路
                            path = []
                            path.append(service['src'])
                            for r in sub_routes:
                                path.append(r['route'][-1][-1])

                            # 分段，确定relay
                            path_slices, path_indexes = self.cut_path(path, sub_routes)

                            if path_slices == 0: # 无法满足约束，不可行，检查下一条域间路径的情况
                                continue

                            # 这里才开始考虑频谱分配，确定了每一段全光路径的可用波长，以及使用了哪一个中继器
                            path_indexes = [sublist[:-1] for sublist in path_indexes]

                            if len(path_slices) == 1:  # 无需中继
                                for index_slice, path_index in enumerate(path_indexes):
                                    path_slice = path_slices[index_slice]

                                    if path_slice[0] == service['src']:  # slice s
                                        available_l = [_ for _ in range(self.L)]  # 存储该段可以使用的频隙，不断取交集更新
                                        for y in range(len(path_index)):
                                            available_l_slice = sub_routes[path_index[y]]['layer']
                                            available_l = list(set(available_l) & set(available_l_slice))
                                            if len(available_l) == 0:  # 这条链路没有可以使用的频隙，报废
                                                break

                                        if len(available_l) == 0:
                                            break
                                        else:
                                            # 更新当前全光路径的频谱资源
                                            for y in range(len(path_index)):
                                                update_index = []
                                                for index_retain, _ in enumerate(
                                                        sub_routes[path_index[y]]['layer']):
                                                    if _ in available_l:
                                                        update_index.append(index_retain)

                                                sub_routes[path_index[y]]['layer'] = available_l
                                        success_flag1 = 1

                            else: # 需中继
                                for index_slice, path_index in enumerate(path_indexes):
                                    path_slice = path_slices[index_slice]

                                    if path_slice[0] == service['src']:  # slice s
                                        available_l = [_ for _ in range(self.L)] # 存储该段可以使用的频隙，不断取交集更新
                                        for y in range(len(path_index)):
                                            available_l_slice = sub_routes[path_index[y]]['layer']
                                            available_l = list(set(available_l) & set(available_l_slice))
                                            if len(available_l) == 0: # 这条链路没有可以使用的频隙，报废
                                                break

                                        if len(available_l) == 0:
                                            break
                                        else:
                                            # 注意这里d_r存储下一段启用的中继器序号
                                            for d_r,relay in enumerate(self.G.nodes[path_slice[-1]]['available relay']):
                                                # if np.sum(relay['f_slot']) > 0: # 中继器有可用频隙
                                                    # available_relay = [w for w,_ in enumerate(relay['f_slot']) if _ != 0]
                                                available_relay = [_ for _ in range(self.L)]
                                                break

                                            # 更新当前全光路径的频谱资源
                                            for y in range(len(path_index)):
                                                update_index = []
                                                for index_retain,_ in enumerate(sub_routes[path_index[y]]['layer']):
                                                    if _ in available_l:
                                                        update_index.append(index_retain)

                                                sub_routes[path_indexes[index_slice][y]]['layer'] = available_l

                                            # 下一段的可用波长以该中继器的可用频隙为基准（这里条件放宽，改为具有任意波长转换能力）
                                            available_l = available_relay


                                    elif path_slice[-1] == service['snk']:  # dis slice
                                        # 除了第一段，其余每一段先更新中继器使用情况
                                        sub_routes[path_indexes[index_slice-1][-1]]['relay_index'] = d_r

                                        for y in range(len(path_index)):
                                            available_l_slice = sub_routes[path_index[y]]['layer']
                                            available_l = list(set(available_l) & set(available_l_slice))
                                            if len(available_l) == 0: # 这条链路没有可以使用的频隙，报废
                                                break

                                        if len(available_l) == 0:
                                            break
                                        else:
                                            success_flag1 = 1 # 频谱分配成功，这条路径路由成功

                                            # 更新当前全光路径的频谱资源
                                            for y in range(len(path_index)):
                                                update_index = []
                                                for index_retain, _ in enumerate(
                                                        sub_routes[path_index[y]]['layer']):
                                                    if _ in available_l:
                                                        update_index.append(index_retain)

                                                sub_routes[path_indexes[index_slice][y]]['layer'] = available_l


                                    else:
                                        # 除了第一段，其余每一段先更新中继器使用情况
                                        sub_routes[path_indexes[index_slice-1][-1]]['relay_index'] = d_r

                                        for y in range(len(path_index)):
                                            available_l_slice = sub_routes[path_index[y]]['layer']
                                            available_l = list(set(available_l) & set(available_l_slice))
                                            if len(available_l) == 0: # 这条链路没有可以使用的频隙，报废
                                                break

                                        if len(available_l) == 0:
                                            break
                                        else:
                                            # 注意这里d_r存储下一段启用的中继器序号
                                            for d_r, relay in enumerate(
                                                    self.G.nodes[path_slice[-1]]['available relay']):
                                                # if np.sum(relay['f_slot']) > 0:  # 中继器有可用频隙
                                                    # available_relay = [w for w, _ in enumerate(relay['f_slot']) if
                                                    #                    _ != 0]
                                                available_relay = [_ for _ in range(self.L)]
                                                break

                                            # 更新当前全光路径的频谱资源
                                            for y in range(len(path_index)):
                                                update_index = []
                                                for index_retain, _ in enumerate(
                                                        sub_routes[path_index[y]]['layer']):
                                                    if _ in available_l:
                                                        update_index.append(index_retain)

                                                sub_routes[path_indexes[index_slice][y]]['layer'] = available_l

                                            # 下一段的可用波长以该中继器的可用频隙为基准
                                            available_l = available_relay

                            if success_flag1:
                                break

                    else:
                        if end_index != -1:
                            # 找出这个路由失败的部分，从域间路径表中剔除，避免重复计算
                            failure_part = domain_route[end_index:end_index+2]
                            L = len(failure_part)

                            # # 从 A 中删除前 L 项等于 B 的元素
                            # domain_routes_flat = list(filter(lambda sublist: sublist[:L] != failure_part, domain_routes_flat))
                            # 从A中删除含B的元素
                            domain_routes_flat = [sublist for sublist in domain_routes_flat if
                                 not self.contains_consecutive_sublist(sublist, failure_part)]

                        continue

                    # end_relay = time.time()
                    # print(f"中继和频谱分配时间 = {end_relay - start_relay}")


                    if success_flag1: # 业务路由成功
                        # self.record_success(index) # 测试的时候把这个注释掉，频繁打开关闭文件会占用时间

                        subroute = self.chosen_route(sub_routes)  # 确定具体的路由路线，这里也许有优化空间，怎么选择？

                        real_success_flag = self.check_success(subroute, service) # 检查业务是否真的成功了，从路径有效性，中继器有效性，频谱有效性三个方面考虑
                        # real_success_flag = 1

                        if real_success_flag:
                            num_succeed += 1
                            end_service = time.time()
                            time_succeed_list.append(end_service - start_service)
                            len_domain_list.extend([len(r) for r in domain_routes])

                            print(f"服务{index}路由成功！")
                            print(f"domain_route = {domain_route}")
                            print(f"服务处理时间 = {end_service - start_service}")
                            print(f"域内处理时间{end_service-start_service-time_domain}")

                            start_update = time.time()
                            self.updata_state(domain_route, subroute)  # 思考一下还有没有必要，如果进行域间路径刷新的话
                            end_update = time.time()
                            print(f"更新时间 = {end_update - start_update}")


                            start_update = time.time()
                            self.update_subgraph(domain_route)  # 更新域级图，对于本次路由涉及到的各个域，检查每个域是否是连通图，这是作为域的基本条件，不满足的话需要将这个域重新聚类，并调整和邻域的关系
                            end_update = time.time()
                            print(f"更新域级图时间(拆解) = {end_update - start_update}")
                            print('\n')

                            break


        END = time.time()
        print(f"总时间 = {END - START}")
        # resource_occupation_after = self.resource_occupation(self.G)
        print('num_succeed:', num_succeed)
        print(f"Service success rate: {num_succeed / len(self.services[:2501])}%")
        print('ave time (success):', np.mean(time_succeed_list))
        print('time indomain:', np.mean(time_indomain_list))
        print('len domain route:', np.mean(len_domain_list))
        print('time for service:', np.mean(time_for_service))
        # print(f"resource occupation before: {resource_occupation_before}, resource occupation after: {resource_occupation_after}")
        # print(len_domain_list)
        print(success_rate_list)
        print(domain_portion_list)
        print(indomain_portion_list)
        # print(f"域间处理最长时间 = {np.max(time_domain_list)}, 最短时间 = {np.min(time_domain_list)}，平均时间 = {np.mean(time_domain_list)}")
        # print(f"域内处理最长时间 = {np.max(time_indomain_list)}, 最短时间 = {np.min(time_indomain_list)}，平均时间 = {np.mean(time_indomain_list)}")




if __name__ == '__main__':
    FILE_ADD = 'RSOP_80/'

    # for example1
    file_name='example1'
    file_path='example/'
    band = 24
    c_max = 964
    subgraph_file_path = './subgraphs/example1_1/'


    # # for example2
    # file_name = 'example2'
    # file_path = 'example/'
    # band = 8
    # c_max = 864
    # subgraph_file_path = './subgraphs/example2_2/'

    # 均匀/不均匀小规模 6/3
    file_name='example3'
    file_path='example/'
    band = 8
    c_max = 968
    subgraph_file_path = './subgraphs/' + FILE_ADD + 'example3_3/'
    #
    # # 均匀/不均匀大规模 5/4
    # file_name='example4'
    # file_path='example/'
    # band = 24
    # c_max = 984
    # subgraph_file_path = './subgraphs/' + FILE_ADD + 'example4_4/'

    distance_margin = 8 *100
    ots_margin = 10 * 1
    osnr_margin = 10 * 0.001

    # # only for example 5
    # distance_margin = 5 * 100
    # ots_margin = 5 * 1
    # osnr_margin = 5 * 0.001




    S = ServiceRecovery(distance_margin, ots_margin, osnr_margin, subgraph_file_path=subgraph_file_path, file_path=file_path, file_name=file_name, band=band, c_max = c_max, FILE_ADD = FILE_ADD)


    # start_time = time.time()
    S.run()
    # end_time = time.time()