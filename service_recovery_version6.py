import random

import networkx as nx
import numpy as np
from cluster_new2 import *
from cluster_new import *
import itertools
import heapq
import pandas as pd
from itertools import islice
from networkx import NetworkXNoPath
from networkx import NetworkXError


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
    def __init__(self, distance_margin, ots_margin, osnr_margin, subgraph_file_path, file_path='example/', file_name='example1', band=24,c_max = 964):
        self.distance_margin = distance_margin
        self.ots_margin = ots_margin
        self.osnr_margin = osnr_margin

        self.file_name = file_name
        self.file_path = file_path

        self.G = create_topology(file_path, file_name, band, c_max)  # 拓扑，原图
        self.G = nx.Graph(self.G) # 变成单重边图
        self.subgraphs = load_sg(subgraph_file_path) # 载入聚类结果

        self.G_sub = self.G_domain() # 将域抽象为节点的子图，体现域间连接

        self.services = process_service(file_path, file_name, band, c_max) # 业务信息
        self.relay_count, self.relay_index = self.get_relay_node() # 原图中继节点信息（数量+索引）


        self.N = self.G.number_of_nodes()  # 节点数量
        self.M = self.G.number_of_edges()  # 边数量
        self.L = self.G.graph['L']
        self.subgraphs_layering = self.creat_sub_layering()  # [k][l] 第k个子图第l层分层辅助图 # 针对每个子图，按照频隙层进行进一步划分，生成分层辅助图

        # 处理节点在哪个域
        self.process_domain()

    @staticmethod
    # 路由计算
    def route(G, s, d, k=1):
        if k == 1:
            try:
                # 尝试获取最短路径长度
                # return nx.dijkstra_path(G, s, d)
                return nx.shortest_path(G, s, d)

            except NetworkXNoPath:
                # 如果无路径，返回 0
                return 0

    def get_relay_node(self):
        filename = self.file_path + self.file_name + '.relay.csv'

        df = pd.read_csv(filename)
        unique_values_count = df['nodeId'].nunique()  # 中继节点的数量
        unique_values = df['nodeId'].unique()  # 所有中继节点的索引
        return unique_values_count, unique_values

    @staticmethod
    def resource_occupation(G):
        free_slot = 0
        num_e = 0
        for u, v in G.edges():
            num_e += 1
            free_slot += sum(G[u][v]['f_slot'])
            L_slot = len(G[u][v]['f_slot'])
        return 1 - free_slot / (num_e * L_slot)

    # 生成分层辅助图 for subgraph
    def creat_g_layering(self, subgraph):
        # network = nx.DiGraph()

        subgraph_layering_list = []
        for l in range(self.L):
            network = nx.MultiGraph()
            network.add_nodes_from(subgraph.nodes())
            for u, v in subgraph.edges():
                edge_list = []
                if self.G[u][v]['f_slot'][l] == 1:  # 在这里考虑了频谱
                    edge_list.append((u, v))
                network.add_edges_from(edge_list)
            subgraph_layering_list.append(network)

        return subgraph_layering_list

    def creat_sub_layering(self):
        # print(self.G.edges())
        # a = self.G[15][332]
        # print(self.G[15][332])
        subgraphs_layering = []  # [k][l] 第k个子图第l层分层辅助图
        for index, subgraph in enumerate(self.subgraphs):
            subgraphs_layering.append(self.creat_g_layering(subgraph))
        return subgraphs_layering

    # 处理节点在哪个域，域间重点不重边
    def process_domain(self):
        for n in self.G.nodes():
            self.G.nodes[n]['domain'] = set()
            for i, sg in enumerate(self.subgraphs):
                if sg.has_node(n):
                    self.G.nodes[n]['domain'].update([i])


    # 更新分层辅助图（部署请求后）
    def update_g_layering(self):
        pass

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
                    G_sub[i][j]['relayable'] = True  # 标记两个子图之间是否存在相同的电中继节点，方便计算潜在的电中继域间路径
                    G_sub[i][j]['available_relay'] = common_relay  # 记录这两个域进行域间传输时，可用的所有中继节点
                else:
                    G_sub[i][j]['relayable'] = False

            # 还需要考虑位于各个域边缘的中继节点之间的连接！
            margin1_relay = [k for k in margin1 if self.G.nodes[k]['relay'] == True]
            margin2_relay = [k for k in margin2 if self.G.nodes[k]['relay'] == True]
            for r1 in margin1_relay:
                for r2 in margin2_relay:
                    if self.G.has_edge(r1, r2):  # 比如example1中的中继节点4的情况
                        # 检查一下这些链接光参值在不在边界值内
                        flag_within,_,_,_ = self.check_margin([r1,r2])
                        if flag_within:
                            common_relay.append(str(r1) + '+' + str(r2))
                            G_sub.add_edge(i, j)
                            G_sub[i][j]['relayable'] = True
                            G_sub[i][j]['available_relay'] = common_relay  # 两个域之间不是通过共同的电中继节点，而是通过边缘电中继节点之间的直连进行连接的情况特别存储！！！

        return G_sub

        # Yen 算法用于找到 K 条最短路径。它在多域路由的情况下使用，以提供多个路径选择
        # 不一定是全局的K条最短啊



    def bidirectional_bfs_with_constraints(self, start, end, src_rest, snk_rest, max_depth):
        """
        双向 BFS 寻找路径，加入约束条件。

        Args:
            start: 起始节点。
            end: 终止节点。
            src_rest: 起始域限制的节点集合。
            max_depth: 最大搜索深度。

        Returns:
            list: 找到的路径，若无解则返回空列表。
        """


        # 初始化正向和反向队列
        forward_queue = [(0, start, [], [-1], 0)]  # (路径长度, 当前节点, 当前路径, prov_overlap, 深度)
        backward_queue = [(0, end, [], [-1], 0)]  # 同上，但反向搜索
        forward_visited = {}  # {node: (路径长度, 当前路径, prov_overlap)}
        backward_visited = {}  # {node: (路径长度, 当前路径, prov_overlap)}

        forward_visited[start] = (0, [start], [-1])
        backward_visited[end] = (0, [end], [-1])

        def expand(queue, source_rest, visited, other_visited, direction):
            """扩展当前队列"""
            path_len, node, path, prov_overlap, depth = heapq.heappop(queue)
            if len(path) == 0:
                path = path + [node]
            else:
                path = path

            # 若深度超过限制，跳过
            if depth >= max_depth:
                return None

            # 遍历邻居节点
            neighbors = list(self.G_sub.neighbors(node))

            # 引入随机性
            random.shuffle(neighbors)

            for neighbor in neighbors:
                if neighbor in source_rest or neighbor in path:  # 跳过限制节点或重复节点
                    continue

                overlaps = self.domain_edge_nodes(node, neighbor)
                overlap_result = [
                    int(x.split('+')[1]) if isinstance(x, str) else x for x in overlaps
                ]
                if set(prov_overlap) & set(overlap_result):  # 检查重叠约束
                    continue

                new_path = path + [neighbor]
                if neighbor in visited:  # 如果节点已经访问过，跳过
                    continue

                # 如果邻居在另一侧队列访问过，则找到路径
                if neighbor in other_visited:
                    other_path = other_visited[neighbor][1]
                    prov_neighobor = other_visited[neighbor][2]

                    if set(prov_neighobor) & set(overlap_result): # 检查重叠约束
                        continue

                    if direction == "forward":
                        # return new_path + other_path[::-1]  # 正向路径 + 反向路径
                        return new_path[:-1] + other_path[::-1] # 正向路径 + 反向路径
                    else:
                        return other_path + new_path[::-1][1:] # 反向路径 + 正向路径

                # 更新当前队列和访问记录
                visited[neighbor] = (path_len + 1, new_path, overlap_result)
                heapq.heappush(queue, (path_len + 1, neighbor, new_path, overlap_result, depth + 1))

            return None

        # 双向搜索主循环
        while forward_queue and backward_queue:

            # 扩展较小的队列
            if len(forward_queue) <= len(backward_queue):
                result = expand(forward_queue, src_rest, forward_visited, backward_visited, "forward")
            else:
                result = expand(backward_queue,snk_rest, backward_visited, forward_visited, "backward")

            # 如果找到路径，返回结果
            if result:
                return result

        # 如果搜索结束没有找到路径，返回空列表
        return []

    # 使用BFS的方法寻找域间路径，可以显著减少路径跳数
    # def domain_route(G, start, end):
    def domain_route(self, service):
        num_of_path = 20 # 找到k条路径即停止查找
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

                    # 用BFS找路径，不一定有必要，而且太慢了，虽然长度数量比较可控
                    src_rest = [item for item in src_domain if item != src]
                    snk_rest = [item for item in snk_domain if item != snk]

                    if self.G_sub.has_edge(src, snk):
                        all_paths = [src, snk]
                        all_all_paths.append([all_paths])

                        count_path += 1
                        if count_path == num_of_path:
                            break

                    # else:
                    # 无论有没有直连路径都算一下
                    # search_start = time.time()
                    # all_paths = self.dijkstra_with_constraints(src, snk, src_rest, snk_rest, 1) # 最后一个参数是最大深度，可调
                    # all_paths = self.bidirectional_bfs_with_constraints(src, snk, src_rest, snk_rest, 2)  # 最后一个参数是最大深度，可调
                    # all_paths = list(islice(nx.shortest_simple_paths(self.G_sub, src, snk), 5))

                    try:  # 用内置BFS函数，速度快，但成功率会低，不知道为啥
                        # all_paths = nx.shortest_path(self.G_sub, src, snk)
                        all_paths = list(islice(nx.shortest_simple_paths(self.G_sub, src, snk), 1))
                    except NetworkXNoPath:
                        all_paths = []

                    if len(all_paths) > 0: # 找到了有效路径
                        count_path += 1
                        # all_all_paths.append([all_paths])
                        all_all_paths.append(all_paths)

                        if count_path == num_of_path:
                            break

                    # search_end = time.time()
                    # print(f"路径{all_paths}搜索时间 = {search_end - search_start}")

        return all_all_paths

    # 域内路由
    def indomain_route(self, subgraph_index, s, d, flag, sub_routes):
        if flag == 4:  # relay_pairs
            return self._handle_non_domain_routing(s, d, sub_routes)

        return self._handle_domain_routing(subgraph_index, s, d, sub_routes)

    # 非域内路由逻辑封装
    def _handle_non_domain_routing(self, s, d, sub_routes):
        available_layer = []
        available_route = []
        indexes = []

        for l in range(self.L):
            flag_within, distance, ots, osnr = self.check_margin([s, d])
            if flag_within:
                available_layer.append(l)
                available_route.append([s, d])
                indexes.append([distance, ots, osnr])

        if available_layer:
            sub_routes.append({
                'route': available_route,
                'layer': available_layer,
                'in_domain': False,
                'indexes': indexes,
                'relay_index': -1
            })
            return 1, sub_routes

        return 0, sub_routes

    # 域内路由逻辑封装
    def _handle_domain_routing(self, subgraph_index, s, d, sub_routes):
        available_layer = []
        available_route = []
        indexes = []

        for l in range(self.L):
            # 提前检查是否有路径
            graph_layer = self.subgraphs_layering[subgraph_index][l]
            # if not nx.has_path(graph_layer, s, d):
            #     continue

            route = self.route(graph_layer, s, d)
            if route == 0:
                continue


            flag_within, distance, ots, osnr = self.check_margin(route)
            if flag_within:
                available_layer.append(l)
                available_route.append(route)
                indexes.append([distance, ots, osnr])

        if available_layer:
            sub_routes.append({
                'route': available_route,
                'layer': available_layer,
                'in_domain': True,
                'indexes': indexes,
                'relay_index': -1
            })
            return 1, sub_routes

        return 0, sub_routes

    def domain_edge_nodes(self, s_domain, d_domain): # 计算所有可以用于域间传输的节点
        connect_situ = self.G_sub.get_edge_data(s_domain, d_domain)  # 获取这条链路的信息
        non_relay_nodes_1 = []
        relay_nodes_1 = []
        # 非中继传输
        if 'available_non_relay' in connect_situ:
            non_relay_nodes_1 = connect_situ['available_non_relay']
        # 中继传输，分为一个或两个中继节点传输
        if connect_situ['relayable'] == True:
            relay_nodes_1 = connect_situ['available_relay']

            # 在这里自动把中继节点对对齐了
            for index_k,k in enumerate(relay_nodes_1):
                if isinstance(k, str):
                    s_mar, d_mar = k.split('+')
                    s_mar = int(s_mar)
                    d_mar = int(d_mar)

                    if s_mar in self.subgraphs[s_domain].nodes() and d_mar in self.subgraphs[d_domain].nodes():
                        k = str(s_mar) + '+' + str(d_mar)
                    elif d_mar in self.subgraphs[s_domain].nodes() and s_mar in self.subgraphs[d_domain].nodes():
                        k = str(d_mar) + '+' + str(s_mar)
                    relay_nodes_1[index_k] = k

        d_sub = non_relay_nodes_1 + relay_nodes_1
        return d_sub

    def updata_spectrum(self, path, l):
        for i in range(len(path) - 1):
            self.G[path[i]][path[i + 1]]['f_slot'][l] = 0

    # 更新网络状态，包括占用频隙和子图中的状态更新
    def updata_state(self, domain_route, sub_routes):
        wasted_relay = [] # 记录失效的中继节点
        # domain_index = 0
        for index_r, r in enumerate(sub_routes):

            if len(list(set(r['route']))) == 1:
                continue
            self.updata_spectrum(r['route'], r['layer'])
            if r['relay_index'] != -1:
                self.G.nodes[r['route'][0]]['available relay'][r['relay_index']]['available'] = False
                self.G.nodes[r['route'][0]]['available relay num'] -= 1

                # 检查中继节点还有没有可用的中继器
                if self.G.nodes[r['route'][0]]['available relay num'] == 0:
                    self.G.nodes[r['route'][0]]['relay'] = False # 相当于变成一个普通节点了
                    wasted_relay.append(r['route'][0])

            route = r['route']
            layer = r['layer']

            domain_within = [self.G.nodes[n]['domain'] for n in route]

            # update graph_layring
            if r['in_domain'] == True:

                domain_index = list(set.intersection(*domain_within))

                for j in domain_index:
                    for i in range(len(route) - 1):

                        if self.subgraphs_layering[j][layer].has_edge(route[i], route[i + 1]):
                            self.subgraphs_layering[j][layer].remove_edge(route[i], route[i + 1])
                            if all(x == 0 for x in self.G[route[i]][route[i+1]]['f_slot']): # 只有当这条连接没有可用频隙时才删除
                                try:
                                    self.subgraphs[j].remove_edge(route[i], route[i + 1])
                                except NetworkXError:
                                    pass

        # 部分中继节点变为无效，处理这部分对域级图的影响
        if len(wasted_relay) > 0:
            for w_r in wasted_relay:
                visited_com = [] # 避免重复访问
                domains = self.G.nodes[w_r]['domain'] # 该原中继节点的域
                for domain in domains:
                    del_neighbors = [] # 要删除的连接
                    for neighbor in self.G_sub.neighbors(domain):
                        com = sorted([domain, neighbor])
                        if com not in visited_com:
                            visited_com.append(com)
                            values = self.G_sub.get_edge_data(domain, neighbor)  # 获取这条链路的信息
                            if 'available_relay' in values:
                                relay_con = values['available_relay']
                                relay_int = [item for item in relay_con if isinstance(item, int)]
                                if w_r in relay_int:
                                    relay_int.remove(w_r)

                                    # 仍有可能作为非中继节点继续用于域间传输
                                    if self.subgraphs[domain].degree(w_r) > 0 and self.subgraphs[neighbor].degree(w_r) > 0:
                                        if 'available_non_relay' in values:
                                            if w_r not in values['available_non_relay']:
                                                values['available_non_relay'].append(w_r)

                                relay_str = [item for item in relay_con if isinstance(item, str)]
                                str_result = [item for item in relay_str if str(w_r) not in item]

                                self.G_sub[domain][neighbor]['available_relay'] = relay_int + str_result # 更新这个失效的中继节点所涉及到的连接

                                if len(self.G_sub[domain][neighbor]['available_relay']) == 0:
                                    # 没有中继连接，也没有非中继连接了，才取消域级链接
                                    if 'available_non_relay' in values:
                                        if len(self.G_sub[domain][neighbor]['available_non_relay']) > 0:
                                            continue
                                    del_neighbors.append((domain, neighbor))

                        else:
                            continue
                    self.G_sub.remove_edges_from(del_neighbors)


    def update_subgraph(self, domain_route): # 更新域级图，对于本次路由涉及到的各个域，检查每个域是否是连通图，这是作为域的基本条件，不满足的话需要将这个域重新聚类，并调整和邻域的关系
        new_subs = []
        to_remove = []
        for i in domain_route:
            if not nx.is_connected(self.subgraphs[i]): # 不是连通图

                # 获取连通子图的节点集合,没什么实际作用，主要是方便调试和验证
                connected_components = nx.connected_components(self.subgraphs[i])

                # 需要好好想想怎么着才需要更新域间路径
                to_remove.append(i)
                new_sub = cluster_modify(self.subgraphs[i], self.G, self.distance_margin * 0.6, self.ots_margin * 0.6, self.osnr_margin * 0.6) # 拆散原域，获得新域
                new_subs.append(new_sub)


        original_list = list(range(len(self.subgraphs)))
        # 扁平化列表
        new_subs = [item for sublist in new_subs for item in
                    (sublist if isinstance(sublist, list) else [sublist])]
        print(f"拆分形成了{len(new_subs)}个新域")

        if len(new_subs) > 0:
            self.subgraphs = [self.subgraphs[item] for item in original_list if item not in to_remove]

            self.subgraphs = self.subgraphs + new_subs

            domain_update_s = time.time()
            self.G_sub = self.G_domain()  # 将域抽象为节点的子图，体现域间连接
            domain_update_e = time.time()
            print(f"域级更新完成，花费时间 = {domain_update_e - domain_update_s}")

            layer_update_s = time.time()
            self.subgraphs_layering = self.creat_sub_layering()  # [k][l] 第k个子图第l层分层辅助图 # 针对每个子图，按照频隙层进行进一步划分，生成分层辅助图
            layer_update_e = time.time()
            print(f"分层辅助图更新完成，花费时间 = {layer_update_e - layer_update_s}")

            domain_ensure_s = time.time()
            self.process_domain()
            domain_ensure_e = time.time()
            print(f"定域更新完成，花费时间 = {domain_ensure_e - domain_ensure_s}")


    def chosen_route(self, sub_routes):
        slices = []
        for j in range(len(sub_routes)):
            if sub_routes[j]['relay_index'] != -1:
                slices.append(j)

        if len(slices) == 0: # 整条路径没有使用电中继，共享同一个可用频隙列表
            layers = sub_routes[-1]['layer']
            chosen_index = 0

            for sr in sub_routes:
                sr['layer'] = layers[chosen_index]
                sr['route'] = sr['route'][chosen_index]
            return sub_routes

        elif len(slices) == 1: # 仅使用了一个中继，分为两端
            j = slices[0]
            path1 = sub_routes[:j]
            path2 = sub_routes[j:]
            paths = [path1, path2]

            for pth in paths:
                layers = pth[-1]['layer']
                chosen_index = 0

                for sr in pth:
                    sr['layer'] = layers[chosen_index]
                    sr['route'] = sr['route'][chosen_index]
            return sub_routes

        else: # > 1 ,使用了两个及以上电中继，分为多段
            for index_j, j in enumerate(slices):
                if index_j == 0: # 开头段
                    path = sub_routes[:j]
                elif index_j == len(slices)-1: # 结尾段
                    path = sub_routes[slices[index_j-1]:]
                else: # 中间段
                    path = sub_routes[slices[index_j-1]:j]

                layers = path[-1]['layer']
                chosen_index = 0

                for sr in path:
                    sr['layer'] = layers[chosen_index]
                    sr['route'] = sr['route'][chosen_index]
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

            # 这里就不用逐频隙的考虑了，毕竟是相同的两点之间的光信号的光参值应该不会有太大差异（吧）

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

    # 按照业务端点的距离处理顺序，越远越考前
    def sorted_services(self, services):
        length_list = []
        for service in services:
            length_list.append(len(nx.shortest_path(self.G, service['src'], service['snk'])))
        # 按数据大小排序并返回原索引
        sorted_indices = [index for index, value in sorted(enumerate(length_list), key=lambda x: x[1], reverse=True)]
        # 根据索引重新排序
        services = [services[i] for i in sorted_indices]
        return services




    def run(self):
        resource_occupation_before = self.resource_occupation(self.G)
        num_succeed = 0
        time_succeed_list = []
        time_indomain_list = []
        time_domain_list = []
        time_for_service = []  # 域间时间+最长域内时间
        len_domain_list = []
        start_total = time.time()
        resource_occupation_before = self.resource_occupation(self.G)

        time_all = 0

        success_rate_list = []
        indomain_portion_list = []
        domain_portion_list = []
        # record_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 1800] # example1
        record_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 1800, 2000, 2500] # example2

        # # 打乱服务顺序，看是否有影响（感觉没什么影响啊。。。）
        # random.shuffle(self.services)

        START = time.time()

        self.services = self.sorted_services(self.services[:2501]) # 处理服务顺序，距离远的放前面
        for index, service in enumerate(self.services):

        # for index, service in enumerate(self.services[:2501]):

            if index in record_list:
                success_rate_list.append(num_succeed/index)
                indomain_portion_list.append(np.mean(time_indomain_list))
                domain_portion_list.append(np.mean(time_domain_list))

            start_service = time.time()

            start_domain = time.time()


            domain_routes = self.domain_route(service)  # 查找域间路径
            # d1 = self.G.nodes[1619]['domain']

            # 扁平化列表
            domain_routes_flat = [item for sublist in domain_routes for item in
                                  (sublist if isinstance(sublist, list) else [sublist])]
            domain_routes_flat = [item for item in domain_routes_flat if len(item) != 0 ]

            # 优先选择路径短的
            domain_routes_flat = sorted(domain_routes_flat, key=len)

            # # 打乱列表
            # random.shuffle(domain_routes_flat)

            end_domain = time.time()
            time_domain = end_domain - start_domain
            time_domain_list.append(time_domain)

            print(f"域间传输时间 = {time_domain}")
            print(f"域间路径个数 = {len(domain_routes_flat)}")
            print("\n")

            # route in G Layering for each domain in each layer (First Fit)
            if len(domain_routes_flat) > 0: # 找到路径，表示域间路由成功
                time_all += time_domain


                for index_domain, domain_route in enumerate(domain_routes_flat):
                    success_flag = 0  # 域内传输成功标志
                    start_routing = time.time()

                    if len(domain_route) == 1: # 不需要inter_domain
                        indomain_success_flag, sub_routes = self.indomain_route(domain_route[0],
                                                                        service['src'],
                                                                        service['snk'],
                                                                        3,
                                                                        sub_routes=[])
                        if indomain_success_flag:# 业务路由成功
                            success_flag = 1
                        else:
                            continue


                    else: # 需要inter_domain

                        # 先找出域内路径，再考虑域间连接方式
                        for i, subgraph_index in enumerate(domain_route):
                            # 源域
                            if i == 0:
                                indomain_success_flag = 0  # 域间传输成功标志
                                s = service['src']  # 当前节点（源节点）
                                s_d = domain_route[i]  # 当前域
                                d_d = domain_route[i + 1]  # 目标域
                                d_sub = self.domain_edge_nodes(s_d, d_d)  # 所有可以用于域间传输的节点，包括（中继， 非中继，’中继+中继‘）

                                # # 在这里引入随机性，怎么让每次打乱的都不一样啊，好像要取不同的随机数种子
                                # random.shuffle(d_sub)

                                for d in d_sub:
                                    if isinstance(d, str):  # 中继节点对
                                        s_mar, d_mar = d.split('+')
                                        s_margin = int(s_mar)
                                        d_margin = int(d_mar)

                                        if s == s_margin: # 一段非域内路径
                                            indomain_success_flag1, sub_routes = self.indomain_route(subgraph_index,
                                                                                                     s_margin, d_margin,
                                                                                                     4, sub_routes=[])
                                            if indomain_success_flag1:
                                                indomain_success_flag = 1
                                            else:
                                                continue

                                        else: # 一段域内路径加一段非域内路径
                                            indomain_success_flag1, sub_routes = self.indomain_route(subgraph_index, s,
                                                                                                     s_margin, 0,
                                                                                                     sub_routes=[])


                                            if indomain_success_flag1:
                                                indomain_success_flag2, sub_routes = self.indomain_route(subgraph_index,
                                                                                                         s_margin,
                                                                                                         d_margin, 4,
                                                                                                         sub_routes)

                                                if indomain_success_flag2:
                                                    indomain_success_flag = 1
                                                else:
                                                    continue
                                            else:
                                                continue

                                    else: # 一段域内路径
                                        indomain_success_flag1, sub_routes = self.indomain_route(subgraph_index, s, d,
                                                                                                 0, sub_routes=[])
                                        if indomain_success_flag1:
                                            indomain_success_flag = 1
                                        else:
                                            continue

                                    if indomain_success_flag:
                                        break

                                if indomain_success_flag:  # find indomain route, next domain
                                    continue
                                else:
                                    break  # can't find route, next domain route

                            # 目的域
                            elif i == len(domain_route) - 1:
                                s = sub_routes[-1]['route'][-1][-1]  # 上个域路由的最后节点
                                d = service['snk']  # 目标节点

                                if s == d:
                                    success_flag = 1
                                    break

                                else:
                                    indomain_success_flag1, sub_routes = self.indomain_route(subgraph_index, s, d, 2,
                                                                                             sub_routes)
                                    if indomain_success_flag1:
                                        success_flag = 1
                                        break

                            # 中间域
                            else:
                                indomain_success_flag = 0

                                s = sub_routes[-1]['route'][-1][-1]

                                s_d = domain_route[i]
                                d_d = domain_route[i + 1]
                                d_sub = self.domain_edge_nodes(s_d, d_d)

                                # 备份一份，方便路由失败及时恢复，把字典转换成列表，不可变数据类型
                                sub_routes_ori = [list(item.items()) for item in sub_routes]

                                # # 在这里引入随机性
                                # random.shuffle(d_sub)

                                for d in d_sub:
                                    if isinstance(d, str):
                                        s_mar, d_mar = d.split('+')
                                        s_margin = int(s_mar)
                                        d_margin = int(d_mar)

                                        if s == s_margin: # 一段非域内路径
                                            sub_routes = [dict(item) for item in sub_routes_ori]

                                            indomain_success_flag1, sub_routes = self.indomain_route(subgraph_index,
                                                                                                     s_margin, d_margin,
                                                                                                     4, sub_routes)
                                            if indomain_success_flag1:
                                                indomain_success_flag = 1
                                            else:
                                                continue

                                        else: # 一段域内路径加一段非域内路径
                                            sub_routes = [dict(item) for item in sub_routes_ori]
                                            indomain_success_flag1, sub_routes = self.indomain_route(subgraph_index, s,
                                                                                                     s_margin, 1,
                                                                                                     sub_routes)

                                            if indomain_success_flag1:
                                                indomain_success_flag2, sub_routes = self.indomain_route(subgraph_index,
                                                                                                         s_margin,
                                                                                                         d_margin, 4,
                                                                                                         sub_routes)

                                                if indomain_success_flag2:
                                                    indomain_success_flag = 1
                                                else:
                                                    continue
                                            else:
                                                continue

                                    else: # 一段域内路径
                                        sub_routes = [dict(item) for item in sub_routes_ori]
                                        indomain_success_flag1, sub_routes = self.indomain_route(subgraph_index, s, d,
                                                                                                 1, sub_routes)
                                        if indomain_success_flag1:
                                            indomain_success_flag = 1
                                        else:
                                            continue

                                    if indomain_success_flag:
                                        break

                                if indomain_success_flag:  # find indomain route, next domain
                                    continue
                                else:
                                    break  # can't find route, next domain route


                    end_routing = time.time()
                    print(f"路由计算时间 = {end_routing - start_routing}")

                    if success_flag: # 对于该条域间连接情况，所有的域内连接都可以成功找到

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
                        if len(path_slices) == 1:  # 无需中继
                            success_flag1 = 0
                            for index_slice, path_slice in enumerate(path_slices):

                                if path_slice[0] == service['src']:  # slice s
                                    available_l = [_ for _ in range(self.L)]  # 存储该段可以使用的频隙，不断取交集更新
                                    for y in range(len(path_slice) - 1):
                                        available_l_slice = sub_routes[path_indexes[index_slice][y]]['layer']
                                        available_l = list(set(available_l) & set(available_l_slice))
                                        if len(available_l) == 0:  # 这条链路没有可以使用的频隙，报废
                                            break

                                    if len(available_l) == 0:
                                        break
                                    else:
                                        # 更新当前全光路径的频谱资源
                                        for y in range(len(path_slice) - 1):
                                            update_index = []
                                            for index_retain, _ in enumerate(
                                                    sub_routes[path_indexes[index_slice][y]]['layer']):
                                                if _ in available_l:
                                                    update_index.append(index_retain)

                                            available_route = [sub_routes[path_indexes[index_slice][y]]['route'][w] for
                                                               w in update_index]
                                            available_index = [sub_routes[path_indexes[index_slice][y]]['indexes'][w]
                                                               for
                                                               w in update_index]

                                            sub_routes[path_indexes[index_slice][y]]['layer'] = available_l
                                            sub_routes[path_indexes[index_slice][y]]['route'] = available_route
                                            sub_routes[path_indexes[index_slice][y]]['indexes'] = available_index

                                    success_flag1 = 1


                        else: # 需中继
                            success_flag1 = 0
                            for index_slice, path_slice in enumerate(path_slices):
                                if path_slice[0] == service['src']:  # slice s
                                    available_l = [_ for _ in range(self.L)] # 存储该段可以使用的频隙，不断取交集更新
                                    for y in range(len(path_slice)-1):
                                        available_l_slice = sub_routes[path_indexes[index_slice][y]]['layer']
                                        available_l = list(set(available_l) & set(available_l_slice))
                                        if len(available_l) == 0: # 这条链路没有可以使用的频隙，报废
                                            break

                                    if len(available_l) == 0:
                                        break
                                    else:
                                        # 注意这里d_r存储下一段启用的中继器序号
                                        for d_r,relay in enumerate(self.G.nodes[path_slice[-1]]['available relay']):
                                            if np.sum(relay['f_slot']) > 0: # 中继器有可用频隙
                                                available_relay = [w for w,_ in enumerate(relay['f_slot']) if _ != 0]
                                                break

                                        # 更新当前全光路径的频谱资源
                                        for y in range(len(path_slice)-1):
                                            update_index = []
                                            for index_retain,_ in enumerate(sub_routes[path_indexes[index_slice][y]]['layer']):
                                                if _ in available_l:
                                                    update_index.append(index_retain)

                                            available_route = [sub_routes[path_indexes[index_slice][y]]['route'][w] for
                                                           w in update_index]
                                            available_index = [sub_routes[path_indexes[index_slice][y]]['indexes'][w] for
                                                           w in update_index]

                                            sub_routes[path_indexes[index_slice][y]]['layer'] = available_l
                                            sub_routes[path_indexes[index_slice][y]]['route'] = available_route
                                            sub_routes[path_indexes[index_slice][y]]['indexes'] = available_index


                                        # 下一段的可用波长以该中继器的可用频隙为基准
                                        available_l = available_relay


                                elif path_slice[-1] == service['snk']:  # dis slice
                                    # 除了第一段，其余每一段先更新中继器使用情况
                                    sub_routes[path_indexes[index_slice][0]]['relay_index'] = d_r

                                    for y in range(len(path_slice)-1):
                                        available_l_slice = sub_routes[path_indexes[index_slice][y]]['layer']
                                        available_l = list(set(available_l) & set(available_l_slice))
                                        if len(available_l) == 0: # 这条链路没有可以使用的频隙，报废
                                            break

                                    if len(available_l) == 0:
                                        break
                                    else:
                                        success_flag1 = 1 # 频谱分配成功，这条路径路由成功

                                        # 更新当前全光路径的频谱资源
                                        for y in range(len(path_slice) - 1):
                                            update_index = []
                                            for index_retain, _ in enumerate(
                                                    sub_routes[path_indexes[index_slice][y]]['layer']):
                                                if _ in available_l:
                                                    update_index.append(index_retain)

                                            available_route = [sub_routes[path_indexes[index_slice][y]]['route'][w]
                                                               for
                                                               w in update_index]
                                            available_index = [
                                                sub_routes[path_indexes[index_slice][y]]['indexes'][w] for
                                                w in update_index]

                                            sub_routes[path_indexes[index_slice][y]]['layer'] = available_l
                                            sub_routes[path_indexes[index_slice][y]]['route'] = available_route
                                            sub_routes[path_indexes[index_slice][y]]['indexes'] = available_index

                                else:
                                    # 除了第一段，其余每一段先更新中继器使用情况
                                    sub_routes[path_indexes[index_slice][0]]['relay_index'] = d_r

                                    for y in range(len(path_slice)-1):
                                        available_l_slice = sub_routes[path_indexes[index_slice][y]]['layer']
                                        available_l = list(set(available_l) & set(available_l_slice))
                                        if len(available_l) == 0: # 这条链路没有可以使用的频隙，报废
                                            break

                                    if len(available_l) == 0:
                                        break
                                    else:
                                        # 注意这里d_r存储下一段启用的中继器序号
                                        for d_r, relay in enumerate(
                                                self.G.nodes[path_slice[-1]]['available relay']):
                                            if np.sum(relay['f_slot']) > 0:  # 中继器有可用频隙
                                                available_relay = [w for w, _ in enumerate(relay['f_slot']) if
                                                                   _ != 0]

                                                break

                                        # 更新当前全光路径的频谱资源
                                        for y in range(len(path_slice) - 1):
                                            update_index = []
                                            for index_retain, _ in enumerate(
                                                    sub_routes[path_indexes[index_slice][y]]['layer']):
                                                if _ in available_l:
                                                    update_index.append(index_retain)

                                            available_route = [sub_routes[path_indexes[index_slice][y]]['route'][w]
                                                               for
                                                               w in update_index]
                                            available_index = [
                                                sub_routes[path_indexes[index_slice][y]]['indexes'][w] for
                                                w in update_index]

                                            sub_routes[path_indexes[index_slice][y]]['layer'] = available_l
                                            sub_routes[path_indexes[index_slice][y]]['route'] = available_route
                                            sub_routes[path_indexes[index_slice][y]]['indexes'] = available_index

                                        # 下一段的可用波长以该中继器的可用频隙为基准
                                        available_l = available_relay

                    else:
                        continue

                    # end_relay = time.time()
                    # print(f"中继和频谱分配时间 = {end_relay - start_relay}")


                    if success_flag1: # 业务路由成功
                        # self.record_success(index) # 测试的时候把这个注释掉，频繁打开关闭文件会占用时间

                        subroute = self.chosen_route(sub_routes)  # 确定具体的路由路线，这里也许有优化空间，怎么选择？

                        # # 验证一下是不是真的成功，验证了，是都成功
                        #
                        # if len(subroute) == 1:
                        #     if subroute[0]['route'][0] != service['src'] or subroute[0]['route'][-1] != service['snk']:
                        #         break # 表示actually业务失败
                        # else:
                        #     first_sub = subroute[0]
                        #     last_sub = subroute[-1]
                        #     if first_sub['route'][0] != service['src'] or last_sub['route'][-1] != service['snk']:
                        #         aw = first_sub['route'][0]
                        #         bw = last_sub['route'][-1]
                        #         break

                        start_update = time.time()
                        self.updata_state(domain_route, subroute)  # 思考一下还有没有必要，如果进行域间路径刷新的话
                        end_update = time.time()
                        print(f"更新时间 = {end_update - start_update}")

                        num_succeed += 1
                        end_service = time.time()
                        time_succeed_list.append(end_service - start_service)
                        len_domain_list.extend([len(r) for r in domain_routes])

                        print(f"服务{index}路由成功！")
                        print(f"domain_route = {domain_route}")
                        print(f"服务处理时间 = {end_service - start_service}")
                        print(f"域内处理时间{end_service-start_service-time_domain}")


                        start_update = time.time()
                        self.update_subgraph(domain_route)  # 更新域级图，对于本次路由涉及到的各个域，检查每个域是否是连通图，这是作为域的基本条件，不满足的话需要将这个域重新聚类，并调整和邻域的关系
                        end_update = time.time()
                        print(f"更新域级图时间(拆解) = {end_update - start_update}")
                        print('\n')

                        break


        END = time.time()
        print(f"总时间 = {END - START}")
        resource_occupation_after = self.resource_occupation(self.G)
        print('num_succeed:', num_succeed)
        print(f"Service success rate: {num_succeed / len(self.services[:2501])}%")
        print('ave time (success):', np.mean(time_succeed_list))
        print('time indomain:', np.mean(time_indomain_list))
        print('len domain route:', np.mean(len_domain_list))
        print('time for service:', np.mean(time_for_service))
        print(f"resource occupation before: {resource_occupation_before}, resource occupation after: {resource_occupation_after}")
        # print(len_domain_list)
        print(success_rate_list)
        print(domain_portion_list)
        print(indomain_portion_list)
        # print(f"域间处理最长时间 = {np.max(time_domain_list)}, 最短时间 = {np.min(time_domain_list)}，平均时间 = {np.mean(time_domain_list)}")
        # print(f"域内处理最长时间 = {np.max(time_indomain_list)}, 最短时间 = {np.min(time_indomain_list)}，平均时间 = {np.mean(time_indomain_list)}")




if __name__ == '__main__':
    # # for example1
    # file_name='example1'
    # file_path='example/'
    # band = 24
    # c_max = 964
    # subgraph_file_path = './subgraphs/example1_1/'


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
    subgraph_file_path = './subgraphs/example3_3/'

    # # 均匀/不均匀大规模 5/4
    # file_name='example4'
    # file_path='example/'
    # band = 24
    # c_max = 984
    # subgraph_file_path = './subgraphs/example4_4/'


    # distance_margin = 10 * 154.7047511
    # ots_margin = 10 * 2.63800905
    # osnr_margin = 10 * 0.00556831

    distance_margin = 8 *100
    ots_margin = 10 * 1
    osnr_margin = 10 * 0.001


    S = ServiceRecovery(distance_margin, ots_margin, osnr_margin, subgraph_file_path=subgraph_file_path, file_path=file_path, file_name=file_name, band=band, c_max = c_max)


    # start_time = time.time()
    S.run()
    # end_time = time.time()