import networkx as nx
import random
import time
import numpy as np
import matplotlib.pyplot as plt

# 这部分代码应该就是基于分层辅助图的RSA算法

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)


seed = 1
set_seed(seed)


class RouteLayering:
    def __init__(self, N:int, M:int, L:int, util_rate=0.2):
        self.N = N  # 节点数量
        self.M = M  # 边数量
        self.L = L  # 频隙（波）数量
        self.util_rate = util_rate  # 网络利用率（占用率）, 网络中每条连接的频隙可用比例

        self.G = self.random_g(self.N, self.M, self.L, self.util_rate)  # 拓扑
        self.G_layering = self.creat_g_layering()  # 分层辅助图
        self.random_add_layer_edge(num=int(self.N/2))

    @staticmethod
    # 随机生成网络拓扑, 频谱1：表示可用，0：已被占用
    def random_g(n, m, L, uti):
        network = nx.DiGraph() # 无多重边有向图
        network.add_nodes_from([i for i in range(n)])
        network.add_edges_from([(u, v, {'delay': random.randint(1, 10), 'osnr_loss': random.randint(1, 5),
                                        'f_slot': random.choices([0, 1], weights=[uti, 1-uti], k=L)})
                                for u, v in nx.dense_gnm_random_graph(n, m, seed=seed).edges()])

        return network

    @staticmethod
    # 画图
    def show_g(G):
        print('number_of_nodes:', nx.number_of_nodes(G))
        print('nodes:', G.nodes())
        print('number_of_edges:', nx.number_of_edges(G))
        print('edges:', G.edges(data=True))
        print('degrees:', nx.degree(G)) # G中所有节点的与之相连的边的数量

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True)

        plt.show()

    @staticmethod
    # 路由计算
    def route(G, s, d, k=1):
        if k == 1:
            return nx.dijkstra_path(G, s, d, weight='osnr_loss') # 基于损耗最小寻找路径


    # 生成分层辅助图
    def creat_g_layering(self):
        network = nx.DiGraph()
        for l in range(self.L): # 有多少个频隙，生成多少个辅助图
            network.add_nodes_from(self.N*l + i for i in range(self.N)) # 节点索引依次递增
            for u, v in self.G.edges():
                if self.G[u][v]['f_slot'][l] == 1:
                    network.add_edge(self.N*l + u, self.N*l + v)

        return network

    # 在分层辅助图中添加层间边（节点n进行光电光转化，中继节点）
    def add_layer_edge(self, n):
        for i in range(self.L):
            for j in range(i+1, self.L):
                u = self.N*i + n
                v = self.N*j + n
                self.G_layering.add_edge(u, v)
                self.G_layering.add_edge(v, u)

    # 随机选取num个节点（光电光转化），并添加层间边
    def random_add_layer_edge(self, num=1):
        node_list = random.choices([i for i in range(self.N)], k=num)
        for n in node_list:
            self.add_layer_edge(n)

    # 更新分层辅助图（部署请求后）
    def update_g_layering(self):
        pass

    # 测试平均路由时间（times次）, 若flag为1，在分层辅助图计算
    def test_route_time(self, times, layer_flag=0):
        T = []
        L = 0
        for i in range(times):
            if layer_flag:
                u = random.randint(0, self.N*self.L - 1) # 两种的区别在于节点索引的范围
                v = random.randint(0, self.N*self.L - 1)
                while not nx.has_path(self.G_layering, u, v):
                    L += 1
                    # print(L)
                    u = random.randint(0, self.N*self.L - 1)
                    v = random.randint(0, self.N*self.L - 1)
                start = time.time()
                # print('start:', start)
                path = self.route(self.G_layering, u, v)
                print('path:', path)
                end = time.time()
                # print('end:', end)
            else:
                u = random.randint(0, self.N - 1)
                v = random.randint(0, self.N - 1)
                while not nx.has_path(self.G, u, v):
                    u = random.randint(0, self.N - 1)
                    v = random.randint(0, self.N - 1)
                start = time.time()
                # print('start:', start)
                path = self.route(self.G, u, v)
                print('path:', path)
                end = time.time()
                # print('end:', end)
            T.append(end - start)
        # print('route calculation time:', end - start)
        print('averange route calculation time:', np.mean(T))


# if __name__ == "__main__":
#     N = 1000
#     M = 5000
#     L = 120
#     Router = RouteLayering(N, M, L)
#     # Router.show_g(Router.G_layering)
#     # fig = Router.random_g(N, M, L, 0.5)
#     # Router.show_g(fig)
#     Router.test_route_time(times=100, layer_flag=1)
#     # Router.show_g(Router.G_layering)

# N = 10
# M = 10
# L = 3
# Router = RouteLayering(N, M, L)
# Router.show_g(Router.G)