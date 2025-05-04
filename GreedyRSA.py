import random
import numpy as np
from cluster import *
from Date_processing import create_topology,process_service


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)


seed = 1
set_seed(seed)


class GreedyRSA():
    def __init__(self, distance_margin, ots_margin, osnr_margin, subgraph_file_path, file_path='example/', file_name='example1', band=24,c_max = 964):
        self.distance_margin = distance_margin
        self.ots_margin = ots_margin
        self.osnr_margin = osnr_margin

        self.file_path = file_path
        self.file_name = file_name
        self.band = band
        self.G = create_topology(file_path, file_name, band, c_max)  # 拓扑
        self.G_normal = self.Multigraph_to_graph(self.G)
        self.services = process_service(file_path, file_name, band,c_max)

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

        return path_list

    @staticmethod
    def KSP(G, source, target, K):
        paths = list(nx.all_shortest_paths(G, source=source, target=target, weight='distance'))
        return paths[:K]

    @staticmethod
    def resource_occupation(G, band):
        free_slot = 0
        num_e = 0
        for u, v in G.edges():
            for e in G[u][v].values():
                num_e += 1
                free_slot += sum(e['f_slot'])
                L_slot = len(e['f_slot'])
        return 1 - free_slot/(num_e*L_slot)


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
            # # 假设G是多重图，我们需要找到正确的边
            # for key, attr in self.G[current_node][next_node].items():
            #     total_distance += attr['distance']
            #     ots += attr['ots']
            #     min_osnr += attr['osnr']

        # 检查总距离、OTS跳数和最小OSNR是否在边际值内
        return (total_distance <= distance_margin and
                ots <= ots_margin and
                total_osnr <= osnr_margin
                          )

    def find_relay(self, path):
        # end_node = path[-1]
        for i in range(len(path)-1, 0, -1):
            if self.check_margin(path[:i+1]) and (i==len(path)-1 or self.G.nodes[path[i]]['relay']):
                return i
        return 0

    def cut_path(self, path):
        start_node = 0
        path_slices = []
        while start_node != len(path)-1:
            end_node = self.find_relay(path[start_node:])
            if end_node == 0:
                return 0  # 无法满足约束
            path_slices.append(path[start_node:start_node+end_node+1])
            start_node = start_node+end_node
        return path_slices

    def if_spectrum_available_for_path(self, path, l):
        edge_index_of_path = []
        for i in range(len(path) - 1):
            for k, e in enumerate(self.G[path[i]][path[i+1]].values()):
                if e['f_slot'][l] == 1:
                    edge_index_of_path.append(k)
                    break
            if len(edge_index_of_path) <= i:  # 所有多重边不可用
                return []
        return edge_index_of_path

        # return all([self.G[path[i]][path[i+1]].values()['f_slot'][l] == 1 for i in range(len(path)-1)])

    # Spectrum and relay allocation
    def spectrum_and_relay_allocation(self, path_slice, s_r, flag):
        if flag == 0:
            f_slot_relay_s = [1 for i in range(self.L)]
        else:
            f_slot_relay_s = self.G.nodes[path_slice[0]]['available relay'][s_r]['f_slot']

        if flag == 2:  # dis slice
            f_slot_relay_d = [1 for i in range(self.L)]
            for l in range(self.L):

                if f_slot_relay_s[l] and f_slot_relay_d[l]:
                    edge_index_of_path = self.if_spectrum_available_for_path(path_slice, l)
                    if edge_index_of_path:
                        return l, -1, edge_index_of_path

        for d_r, relay in enumerate(self.G.nodes[path_slice[-1]]['available relay']):
            if not relay['available']:
                continue
            f_slot_relay_d = relay['f_slot']
            for l in range(self.L):
                if f_slot_relay_s[l] and f_slot_relay_d[l]:
                    edge_index_of_path = self.if_spectrum_available_for_path(path_slice, l)
                    if edge_index_of_path:
                        return l,  d_r, edge_index_of_path
        return -1, -1, []

    def updata_spectrum(self, path, edge_index_of_path, l):
        for i in range(len(path) - 1):
            self.G[path[i]][path[i + 1]][edge_index_of_path[i]]['f_slot'][l] = 0

    def updata_state(self, sub_routes):
        for r in sub_routes:
            self.updata_spectrum(r['route'], r['edge_index_of_path'], r['layer'])
            if r['relay_index'] != -1:
                self.G.nodes[r['route'][-1]]['available relay'][r['relay_index']]['available'] = False

    def record_success(self, succ_num):
        with open('success_index.txt', 'a') as f:
            f.write(str(succ_num))
            f.write(' ')
            f.close()

    def RSA(self):
        # 初始化计数器
        service_success_count = 0  # 成功业务计数
        service_total_count = 0  # 总业务计数
        service_success_route_recorder = []

        path_length = [] # 记录路由长度

        resource_occupation_before = self.resource_occupation(self.G, self.band)

        x = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 1800, 2000, 2500]
        success_rate_list = []
        resource_uti_list = []

        start_all = time.time()
        # 读取业务需求并处理
        for index, service in enumerate(self.services[:2501]):
            if index in x:
                success_rate_list.append(service_success_count/index)
                resource_uti_list.append(self.resource_occupation(self.G, self.band))

            service_total_count += 1
            if service_total_count == 401:
                bb=1
            # 寻找k条最短路径
            k = 15  # 默认k值
            # paths1 = self.YenKSP(self.G, service['src'], service['snk'], K=k)
            paths = self.YenKSP(self.G_normal, service['src'], service['snk'], K=k)
            # 打印找到的k个最短路径
            if paths:

                print(
                    f"For service {service_total_count} of {len(self.services)} from {service['src']} to {service['snk']}, found {len(paths)} shortest paths:")
                # for idx, path in enumerate(paths, 1):
                #     print(f"  Path {idx}: {path}")
                # for idx, path in enumerate(paths1, 1):
                #     print(f"  Path_Yen {idx}: {path}")
            else:
                print(f"No paths found for service {service_total_count} from {service['src']} to {service['snk']}.")

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
                        edge_index_of_path = self.if_spectrum_available_for_path(path, l)
                        if edge_index_of_path:
                            self.updata_spectrum(path, edge_index_of_path, l)
                            service_success_count += 1
                            success_flag = 1
                            service_success_route_recorder.append([{'route': path, 'layer': l, 'relay_index': -1,
                                                                    'edge_index_of_path': edge_index_of_path}])
                            print('success')
                            print(f"路径 = {path}")
                            break
                    if success_flag:
                        path_length.append(len(path))

                        self.record_success(service_total_count-1)

                        break

                else:  # 需中继
                    sub_routes = []
                    success_flag = 0
                    for path_slice in path_slices:
                        if path_slice[0] == service['src']:  # slice s
                            l, d_r, edge_index_of_path = self.spectrum_and_relay_allocation(path_slice, 0, flag=0)
                            if l != -1:  # 成功
                                sub_routes.append({'route': path_slice, 'layer': l, 'relay_index': d_r, 'edge_index_of_path': edge_index_of_path})
                            else:  # 失败
                                # print('no enough spectrum or relay')
                                break

                        elif path_slice[-1] == service['snk']:  # dis slice
                            l, d_r, edge_index_of_path = self.spectrum_and_relay_allocation(path_slice, sub_routes[-1]['relay_index'], flag=2)
                            if l != -1:  # 成功
                                sub_routes.append({'route': path_slice, 'layer': l, 'relay_index': d_r, 'edge_index_of_path': edge_index_of_path})
                                success_flag = 1
                            else:  # 失败
                                # print('no enough spectrum or relay')
                                break

                        else:
                            l, d_r, edge_index_of_path = self.spectrum_and_relay_allocation(path_slice, sub_routes[-1]['relay_index'],
                                                                        flag=1)
                            if l != -1:  # 成功
                                sub_routes.append({'route': path_slice, 'layer': l, 'relay_index': d_r, 'edge_index_of_path': edge_index_of_path})
                            else:  # 失败
                                # print('no enough spectrum or relay')
                                break

                    if success_flag:

                        self.record_success(service_total_count - 1)

                        # # update G
                        self.updata_state(sub_routes)
                        # service_success_route_recorder.append(sub_routes)
                        service_success_count += 1
                        print('success')
                        print(f"路径 = {path}")

                        length = sum([len(item) for item in path_slices])
                        path_length.append(length)

                        break

        end_all = time.time()

        resource_occupation_after = self.resource_occupation(self.G, self.band)

        # 计算服务成功率
        service_success_rate = (service_success_count / service_total_count) * 100 if service_total_count > 0 else 0
        print(f"Total services processed: {service_total_count}")
        print(f"Services succeeded: {service_success_count}")
        print(f"Service success rate: {service_success_rate:.2f}%")
        save_pkl(self.file_path + '/' + self.file_name + 'service_success_route_recorder' + '.pkl', service_success_route_recorder)

        print('average time:', (end_all - start_all)/service_total_count)
        print(f"resource occupation before: {resource_occupation_before}, resource occupation after: {resource_occupation_after}")

        print(f"平均路径长度 = {np.mean(path_length)}")

        print(success_rate_list)


if __name__ == '__main__':
    # # for example1
    distance_margin = 800
    ots_margin = 10
    osnr_margin = 0.01

    # for example2
    # distance_margin = 10 * 154.7047511
    # ots_margin = 10 * 2.63800905
    # osnr_margin = 10 * 0.00556831

    Greedy = GreedyRSA(distance_margin, ots_margin, osnr_margin, subgraph_file_path='./subgraphs/' + 'example1' + '/', file_path='example/', file_name='example1', band=24, c_max = 964)

    # Greedy = GreedyRSA(distance_margin, ots_margin, osnr_margin, subgraph_file_path='./subgraphs/' + 'example2' + '/', file_path='example/', file_name='example2', band=8, c_max = 864)
    #
    # Greedy = GreedyRSA(distance_margin, ots_margin, osnr_margin, subgraph_file_path='./subgraphs/' + 'example6' + '/',
    #                    file_path='example/', file_name='example6', band=8, c_max=968)

    # Greedy = GreedyRSA(distance_margin, ots_margin, osnr_margin, subgraph_file_path='./subgraphs/' + 'example4' + '/',
    #                    file_path='example/', file_name='example4', band=24, c_max=984)

    Greedy.RSA()


