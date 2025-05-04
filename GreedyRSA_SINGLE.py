import random
import numpy as np
from cluster import *
from Date_processing import create_topology, process_service
from itertools import groupby
from networkx import NetworkXNoPath
from itertools import islice


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)


seed = 1
set_seed(seed)


class GreedyRSA():
    def __init__(self, distance_margin, ots_margin, osnr_margin, subgraph_file_path, file_path='example/',
                 file_name='example1', band=24, c_max=964, FILE_ADD='RSOP_80/',**kwargs):
        self.distance_margin = distance_margin
        self.ots_margin = ots_margin
        self.osnr_margin = osnr_margin

        self.file_path = file_path
        self.file_name = file_name
        self.band = band
        self.G = create_topology(file_path + FILE_ADD, file_name, band, c_max)  # 拓扑
        self.G_normal = nx.Graph(self.G)
        self.services = process_service(file_path + FILE_ADD, file_name, band, c_max)

        self.N = self.G.number_of_nodes()  # 节点数量
        self.M = self.G.number_of_edges()  # 边数量
        self.L = self.G.graph['L']

    @staticmethod
    def Multigraph_to_graph(G):
        G_n = nx.Graph()
        G_n.add_nodes_from(G.nodes())
        edge_list = []
        for u, v in G.edges():
            edge_list.append((u, v, {'cost': max([e['cost'] for e in G[u][v].values()]),
                                     'distance': max([e['distance'] for e in G[u][v].values()]),
                                     'ots': max([e['ots'] for e in G[u][v].values()]),
                                     'osnr': max([e['osnr'] for e in G[u][v].values()])}
                              ))
        G_n.add_edges_from(edge_list)
        return G_n

    @staticmethod
    def YenKSP(G, source, target, K):
        try:
            path_list = []
            path_list.append(nx.dijkstra_path(G, source, target, weight='weight'))

            for k in range(K - 1):
                temp_path = []
                for i in range(len(path_list[k]) - 1):
                    tempG = G.copy()  # 复制一份图 供删减操作
                    spurNode = path_list[k][i]
                    rootpath = path_list[k][:i + 1]
                    len_rootpath = nx.dijkstra_path_length(tempG, source, spurNode, weight='weight')

                    for p in path_list:
                        if rootpath == p[:i + 1]:
                            if tempG.has_edge(p[i], p[i + 1]):
                                tempG.remove_edge(p[i], p[i + 1])  # 防止与祖先状态重复
                    tempG.remove_nodes_from(rootpath[:-1])  # 防止出现环路
                    if not (nx.has_path(tempG, spurNode, target)):
                        continue  # 如果无法联通，跳过该偏移路径

                    spurpath = nx.dijkstra_path(tempG, spurNode, target, weight='weight')
                    len_spurpath = nx.dijkstra_path_length(tempG, spurNode, target, weight='weight')

                    totalpath = rootpath[:-1] + spurpath
                    len_totalpath = len_rootpath + len_spurpath
                    temp_path.append([totalpath, len_totalpath])
                if len(temp_path) == 0:
                    break

                temp_path.sort(key=(lambda x: [x[1], len(x[0])]))  # 按路程长度为第一关键字，节点数为第二关键字，升序排列
                path_list.append(temp_path[0][0])
        except:
            path_list = []

        return path_list

    @staticmethod
    def KSP(G, source, target, K):
        paths = list(nx.all_shortest_paths(G, source=source, target=target, weight='distance'))
        return paths[:K]


    def resource_occupation(self, G):
        free_slot = 0
        num_e = 0
        for u, v in G.edges():
            for e in G[u][v].values():
                num_e += 1
                free_slot += len(e['f_slot'])

        return 1 - free_slot / (num_e * self.L)

    # 检查链路上的中继节点是否满足边际值要求的函数
    def check_margin(self, path):
        total_distance = 0
        ots = 0
        total_osnr = 0

        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]
            total_distance += min([e['distance'] for e in self.G[current_node][next_node].values()])
            ots += min([e['ots'] for e in self.G[current_node][next_node].values()])
            total_osnr += min([e['osnr'] for e in self.G[current_node][next_node].values()])

        # 检查总距离、OTS跳数和最小OSNR是否在边际值内
        return (total_distance <= self.distance_margin and
                ots <= self.ots_margin and
                total_osnr <= self.osnr_margin
                )

    def find_relay(self, path):
        # end_node = path[-1]
        for i in range(len(path) - 1, 0, -1):
            if self.check_margin(path[:i + 1]) and (i == len(path) - 1 or self.G.nodes[path[i]]['relay']):
                return i
        return 0

    def cut_path(self, path):
        start_node = 0
        path_slices = []
        while start_node != len(path) - 1:
            end_node = self.find_relay(path[start_node:])
            if end_node == 0:
                return 0  # 无法满足约束
            path_slices.append(path[start_node:start_node + end_node + 1])
            start_node = start_node + end_node
        return path_slices

    def if_spectrum_available_for_path(self, path, l):
        path_available = True
        for i in range(len(path) - 1):
            if l not in self.G[path[i]][path[i + 1]][0]['f_slot']:
                path_available = False
                break
        return path_available

        # return all([self.G[path[i]][path[i+1]].values()['f_slot'][l] == 1 for i in range(len(path)-1)])

    # Spectrum and relay allocation
    def spectrum_and_relay_allocation(self, path_slice, s_r, flag):
        # if flag == 0:
        #     f_slot_relay_s = [1 for _ in range(self.L)]
        # else:
        #     f_slot_relay_s = self.G.nodes[path_slice[0]]['available relay'][s_r]['f_slot']

        # 这里简化问题了，中继节点上中继器无限，且具有全波长转换能力

        if flag == 2:  # dis slice
            # f_slot_relay_d = [_ for i in range(self.L)]
            for l in range(self.L):
                # if f_slot_relay_s[l] and f_slot_relay_d[l]:
                flag_available = self.if_spectrum_available_for_path(path_slice, l)
                if flag_available:
                    return l, -1

        for d_r, relay in enumerate(self.G.nodes[path_slice[-1]]['available relay']):
            if not relay['available']:
                continue

            # f_slot_relay_d = relay['f_slot']
            # 改为具有全波长转换能力
            # f_slot_relay_d = [_ for i in range(self.L)]

            for l in range(self.L):
                # if f_slot_relay_s[l] and f_slot_relay_d[l]:
                flag_available = self.if_spectrum_available_for_path(path_slice, l)
                if flag_available:
                    return l, d_r
        return -1, -1

    def updata_spectrum(self, path, l):
        for i in range(len(path) - 1):
            # self.G[path[i]][path[i + 1]][0]['f_slot'][l] = 0
            self.G[path[i]][path[i + 1]][0]['f_slot'].remove(l)

    def updata_state(self, sub_routes):
        for r in sub_routes:
            self.updata_spectrum(r['route'], r['layer'])

            # 改为中继节点上中继器数量无限
            # if r['relay_index'] != -1:
            #     self.G.nodes[r['route'][-1]]['available relay'][r['relay_index']]['available'] = False

    def record_success(self, succ_num):
        with open('success_index.txt', 'a') as f:
            f.write(str(succ_num))
            f.write(' ')
            f.close()

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

        all_shortest_length = np.mean(length_list)

        # 分组相等的元素
        sorted_list.sort(key=lambda x: x[1])
        # 根据第二项分组，结果只保留第一项
        grouped = {key: [item[0] for item in group] for key, group in groupby(sorted_list, key=lambda x: x[1])}
        group_keys = sorted(list(grouped.keys()), reverse=True)

        sorted_indices = []
        # 对于距离相等的业务，根据其路径上的邻居节点平均值/最小值进一步细分，越小越考前
        for group in group_keys:
            service_same_dis = grouped[group]  # 距离相等的业务集合
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

                length_list.append(neighbor_along_path)  # 一路上邻居节点最小值最小值（效果更好）

            sorted_indices1 = [service_same_dis[index] for index, value in
                               sorted(enumerate(length_list), key=lambda x: x[1], reverse=False)]
            sorted_indices += sorted_indices1

        # 根据索引重新排序
        services = [services[i] for i in sorted_indices]  # 按照阻塞概率排列的，越容易阻塞的越靠前
        return services

    def RSA(self):
        # 初始化计数器
        service_success_count = 0  # 成功业务计数
        service_total_count = 0  # 总业务计数
        service_success_route_recorder = []

        path_length = []  # 记录路由长度

        # resource_occupation_before = self.resource_occupation(self.G, self.band)

        x = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 1800, 2000, 2500]
        success_rate_list = []
        resource_uti_list = []

        start_all = time.time()
        # 读取业务需求并处理
        self.services = self.sorted_services(self.services)  # 处理服务顺序，距离远的放前面
        random.shuffle(self.services)
        for index, service in enumerate(self.services):
            if index in x:
                success_rate_list.append(service_success_count / index)
                # resource_uti_list.append(self.resource_occupation(self.G, self.band))

            service_total_count += 1
            if service_total_count == 401:
                bb = 1
            # 寻找k条最短路径
            k = 10  # 默认k值
            # paths1 = self.YenKSP(self.G, service['src'], service['snk'], K=k)
            # paths = self.YenKSP(self.G_normal, service['src'], service['snk'], K=k)
            try:
                paths = list(islice(nx.shortest_simple_paths(self.G_normal, service['src'], service['snk']), k))
            except NetworkXNoPath:
                paths = []

            # 打印找到的k个最短路径
            # if paths:
            #
            #     print(
            #         f"For service {service_total_count} of {len(self.services)} from {service['src']} to {service['snk']}, found {len(paths)} shortest paths:")
            #     # for idx, path in enumerate(paths, 1):
            #     #     print(f"  Path {idx}: {path}")
            #     # for idx, path in enumerate(paths1, 1):
            #     #     print(f"  Path_Yen {idx}: {path}")
            # else:
            #     print(f"No paths found for service {service_total_count} from {service['src']} to {service['snk']}.")

            # if service_total_count == 250:
            #     a = 1

            # 检查每条路径是否满足业务需求
            for path in paths:
                # 分段，确定relay
                path_slices = self.cut_path(path)
                if path_slices == 0:  # 无法满足约束
                    # print('can not slice')
                    continue

                if len(path_slices) == 1:  # 无需中继
                    success_flag = 0
                    for l in range(self.L):
                        flag_available = self.if_spectrum_available_for_path(path, l)
                        if flag_available:
                            self.updata_spectrum(path, l)
                            service_success_count += 1
                            success_flag = 1
                            service_success_route_recorder.append([{'route': path, 'layer': l, 'relay_index': -1}])
                            # print('success')
                            # print(f"路径 = {path}")
                            break
                    if success_flag:
                        path_length.append(len(path))

                        self.record_success(service_total_count - 1)

                        break

                else:  # 需中继
                    sub_routes = []
                    success_flag = 0
                    for path_slice in path_slices:
                        if path_slice[0] == service['src']:  # slice s
                            l, d_r = self.spectrum_and_relay_allocation(path_slice, 0, flag=0)
                            if l != -1:  # 成功
                                sub_routes.append({'route': path_slice, 'layer': l, 'relay_index': d_r})
                            else:  # 失败
                                # print('no enough spectrum or relay')
                                break

                        elif path_slice[-1] == service['snk']:  # dis slice
                            l, d_r = self.spectrum_and_relay_allocation(path_slice, sub_routes[-1]['relay_index'],
                                                                        flag=2)
                            if l != -1:  # 成功
                                sub_routes.append({'route': path_slice, 'layer': l, 'relay_index': d_r})
                                success_flag = 1
                            else:  # 失败
                                # print('no enough spectrum or relay')
                                break

                        else:
                            l, d_r = self.spectrum_and_relay_allocation(path_slice, sub_routes[-1]['relay_index'],
                                                                        flag=1)
                            if l != -1:  # 成功
                                sub_routes.append({'route': path_slice, 'layer': l, 'relay_index': d_r})
                            else:  # 失败
                                # print('no enough spectrum or relay')
                                break

                    if success_flag:
                        # self.record_success(service_total_count - 1)

                        # # update G
                        self.updata_state(sub_routes)
                        # service_success_route_recorder.append(sub_routes)
                        service_success_count += 1
                        # print('success')
                        # print(f"路径 = {path}")

                        length = sum([len(item) for item in path_slices])
                        path_length.append(length)

                        break

        end_all = time.time()

        resource_occupation_after = self.resource_occupation(self.G)

        # 计算服务成功率
        service_success_rate = (service_success_count / service_total_count) * 100 if service_total_count > 0 else 0
        # print(f"Total services processed: {service_total_count}")
        # print(f"Services succeeded: {service_success_count}")
        # print(f"Service success rate: {service_success_rate:.2f}%")
        # save_pkl(self.file_path + '/' + self.file_name + 'service_success_route_recorder' + '.pkl',
        #          service_success_route_recorder)
        #
        # print('average time:', (end_all - start_all) / service_total_count)
        # # print(f"resource occupation before: {resource_occupation_before}, resource occupation after: {resource_occupation_after}")
        #
        # print(f"平均路径长度 = {np.mean(path_length)}")
        #
        # print(success_rate_list)
        return service_success_count, service_total_count, (end_all - start_all) / service_total_count, resource_occupation_after


if __name__ == '__main__':
    FILE_ADD = 'RSOP_20/'

    # # for example1
    distance_margin = 1000
    ots_margin = 10
    osnr_margin = 0.01

    # # for example2
    # distance_margin = 5 * 100
    # ots_margin = 5 * 1
    # osnr_margin = 5 * 0.001

    # Greedy = GreedyRSA(distance_margin, ots_margin, osnr_margin, subgraph_file_path='./subgraphs/' + 'example1' + '/', file_path='example/', file_name='example1', band=24, c_max = 964)
    #
    # Greedy = GreedyRSA(distance_margin, ots_margin, osnr_margin, subgraph_file_path='./subgraphs/' + 'example2' + '/', file_path='example/', file_name='example2', band=8, c_max = 864)
    # # #
    # Greedy = GreedyRSA(distance_margin, ots_margin, osnr_margin, subgraph_file_path='./subgraphs/' + FILE_ADD + 'example7' + '/',
    #                    file_path='example/', file_name='example7', band=8, c_max=968,FILE_ADD=FILE_ADD)
    # #
    Greedy = GreedyRSA(distance_margin, ots_margin, osnr_margin,
                       subgraph_file_path='./subgraphs/' + FILE_ADD + 'example3' + '/',
                       file_path='example/', file_name='example3', band=8, c_max=968, FILE_ADD=FILE_ADD)
    #

    Greedy.RSA()
