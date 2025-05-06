import json
import numpy as np
import networkx as nx
import math
import random
import pandas as pd
from itertools import islice
from itertools import combinations

class Gabriel_Generator():
    def __init__(self, num_nodes, seed, c_max, band, target_occupacy, file_name, num_service, relay_ratio):
        self.num_nodes = num_nodes
        self.seed = seed
        self.c_max = c_max
        self.band = band
        self.target_occupacy = target_occupacy
        self.file_name = file_name
        self.num_service = num_service
        self.relay_ratio = relay_ratio

    def generate_gabriel_graph(self, points):
        """
        根据给定的二维点集生成 Gabriel 图。
        Gabriel 条件：对任意两点 p 和 q，
          如果以 p 和 q 为直径的圆内不包含其他任何点，则在 p 与 q 之间添加边。

        参数:
          points: (n,2) 的 NumPy 数组，每行表示一个点的 [x, y] 坐标。

        返回:
          G: 一个 NetworkX 无向图，节点带有属性 'pos'（坐标，列表形式），边带有 'distance' 属性（欧氏距离）。
        """
        n = len(points)
        G = nx.Graph()
        # 添加节点（节点 ID 为 "R0", "R1", ...）
        for i, pos in enumerate(points):
            G.add_node(i, pos=pos.tolist())

        # 对于每一对节点，检查 Gabriel 条件
        for i in range(n):
            for j in range(i + 1, n):
                p = points[i]
                q = points[j]
                # 计算 p 和 q 的中点 m 以及半径平方 r_sq = ||p-m||^2
                m = (p + q) / 2
                r_sq = np.sum((p - m) ** 2)
                valid = True
                # 检查其它所有点是否落在以 p,q 为直径的圆内
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if np.sum((points[k] - m) ** 2) < r_sq:
                        valid = False
                        break
                if valid:
                    # 满足条件则添加边，同时记录欧氏距离
                    distance = np.linalg.norm(p - q)
                    G.add_edge(i, j, distance=distance)
        return G

    def json_to_csv(self, json_name):
        # 读取 JSON 文件
        with open(json_name, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 提取节点数据
        nodes = data.get("nodes", [])
        # 如果节点的坐标存储在 pos 列表中，可以将其拆分为 x 和 y 两列
        nodes_df = pd.DataFrame(nodes)
        if "pos" in nodes_df.columns:
            nodes_df[["x", "y"]] = pd.DataFrame(nodes_df["pos"].tolist(), index=nodes_df.index)
            nodes_df = nodes_df.drop(columns=["pos"])

        # 将列名 "id" 修改为 "nodeId"
        nodes_df = nodes_df.rename(columns={"id": "nodeId"})

        # 保存节点信息到 Excel 或 CSV 文件
        nodes_df.to_csv("new_example/" + self.file_name + ".node.csv", index=False)

        # 提取链路数据（假设 JSON 中有 "links" 字段）
        links = data.get("links", [])
        links_df = pd.DataFrame(links)

        # 保存链路信息到 Excel 或 CSV 文件
        links_df.to_csv("new_example/" + self.file_name + ".oms_processed.csv", index=False)


    # 生成指定数量的中继节点
    def generate_relays_to_csv(self, G):
        # 生成指定数量个中继节点，根据介数中心性选取
        # 介数中心性衡量一个节点在所有节点对之间最短路径中出现的频率，数值越高通常说明该节点越“关键”或“瓶颈”。
        num_relay = int(len(G.nodes()) * self.relay_ratio)
        # 计算介数中心性
        betweenness = nx.betweenness_centrality(G)
        # 根据介数中心性降序排序节点
        sorted_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        # 选取介数中心性最高的 num_relay 个节点作为中继节点
        relay_nodes = [node for node, centrality in sorted_nodes[:num_relay]]

        # 直接构造 DataFrame，并添加索引列
        df = pd.DataFrame(relay_nodes, columns=['nodeId'])
        # 将 DataFrame 写入 Excel 文件
        df.to_csv('new_example/' + self.file_name + '.relay.csv', index=False)
        print("中继节点CSV文件已生成：output.csv")


    # 生成指定数量的服务请求
    def generate_services_to_csv(self, G):
        """
        从图G中生成num_service个请求，要求请求的源节点和目的节点不同，
        并且它们之间的最短路径长度尽可能长。对于大图，我们采用随机采样部分节点计算距离。

        参数：
            G: NetworkX图对象
            num_service: 要生成的请求数量
            sample_ratio: 采样比例（默认10%）

        返回：
            一个列表，每个元素是字典 {"src": 源节点, "snk": 目的节点, "distance": 最短路径距离}
        """
        # 获取所有节点列表
        nodes = list(G.nodes())
        ser_repeats = {node_id: random.randint(1, 100) for node_id in nodes}  # 每个 nodeId 重复次数范围 (2~16)
        ser_edges = []
        for s_src in ser_repeats.keys():
            s_snk_list = [node for node in ser_repeats.keys() if node != s_src]  # 目的节点不能是源节点
            for _ in range(ser_repeats[s_src]):  # 根据 s_src 的重复次数生成边
                s_snk = random.choice(s_snk_list)  # 随机选择一个 snk
                ser_edges.append((s_src, s_snk))
        random.shuffle(ser_edges)
        ser_edges = ser_edges[:15000] # 最多生成15000个业务请求
        if len(ser_edges) < self.num_service: # 生成的业务请求不够
            print(f"生成的业务请求不够15000个，需要调整参数")
            return
        # 取前 num_service 个请求
        requests = [{"src": pair[0], "snk": pair[1]}
                    for pair in ser_edges]

        # 直接构造 DataFrame，并添加索引列
        df = pd.DataFrame(requests, columns=['src', 'snk'])
        # 将索引重置为一列，并命名为 "Index"
        df.insert(0, 'Index', df.index)

        # 将 DataFrame 写入 Excel 文件
        df.to_csv('new_example/'+self.file_name+'.service.csv', index=False)
        print("业务CSV文件已生成：output.csv")

    # 频谱范围[0,960]， L=120为频隙（波）数量
    # rsop:目标资源占用率
    def generate_colors(self):
        total_slots = math.floor(self.c_max / self.band)
        ranges = [f"{i}-{i + self.band}" for i in range(0, self.c_max, self.band)]

        used_slots = int(total_slots * (1 - self.target_occupacy))  # 计算未被占用的频谱数

        selected_ranges = random.sample(ranges, k=used_slots)  # 随机选择0到120个范围,可以出现空频谱

        # 合并连续的频谱范围
        merged_ranges = []
        selected_ranges.sort(key=lambda x: int(x.split('-')[0]))  # 按照起始值排序
        current_start, current_end = None, None
        for r in selected_ranges:
            start, end = map(int, r.split('-'))
            if current_start is None:
                current_start, current_end = start, end
            elif start <= current_end + 1:  # 连续或重叠
                current_end = max(current_end, end)
            else:
                merged_ranges.append(f"{current_start}-{current_end}")
                current_start, current_end = start, end

        if current_start is not None:  # 添加最后一个范围
            merged_ranges.append(f"{current_start}-{current_end}")

        return ":" + ":".join(merged_ranges)

    # 根据distance计算ots跳数
    def calculate_ots(self, distance, max_distance):
        ots = 0
        if distance <= 100/408 * max_distance:
            ots = 1
        elif distance <= 150/408 * max_distance:
            ots = 2
        elif distance <= 161/408 * max_distance:
            ots = random.choice([2, 3])
        elif distance <= 196/408 * max_distance:
            ots = 3
        elif distance <= 255/408 * max_distance:
            ots = random.choice([3, 4])
        elif distance <= 320/408 * max_distance:
            ots = random.choice([4, 5])
        elif distance <= 340/408 * max_distance:
            ots = 5
        elif distance <= 374/408 * max_distance:
            ots = random.choice([5, 6])
        elif distance <= 400/408 * max_distance:
            ots = 6
        return ots

    # 根据distance计算osnr
    def calculate_osnr(self, distance, max_distance):
        min_value = (0.79809526e-05 * distance - 1.57785657e-04) / 408 * max_distance + random.uniform(0,1.5e-4)
        max_value = (2.58138093e-05 * distance + 2.47357677e-04) / 408 * max_distance + random.uniform(0,1.5e-4)
        osnr = random.uniform(min_value, max_value)
        osnr = max(1e-05, osnr) # 最小为1e-05
        return osnr

    def main(self):
        # 固定随机种子以保证重现性
        np.random.seed(self.seed)
        # 随机生成节点坐标，分布在 [0, 2500]×[0, 2500]
        points = np.random.uniform(0, self.num_nodes * 5, (self.num_nodes, 2))

        # 生成 Gabriel 图
        G = self.generate_gabriel_graph(points)

        # 构造 JSON 数据结构
        data = {
            "graph": {"name": str(num_nodes), "demands": {}},
            "nodes": [],
            "links": []
        }

        # 节点列表：每个节点包含 id 和 pos
        for node, attr in G.nodes(data=True):
            data["nodes"].append({
                "id": node,
                "pos": attr["pos"]
            })

        max_distance = max(G[u][v]['distance'] for u, v in G.edges) # distance最大值

        # 边列表：每条边记录 source, target, distance, ots, osnr, colors
        for u, v, attr in G.edges(data=True):
            ots = self.calculate_ots(attr["distance"], max_distance)
            osnr = self.calculate_osnr(attr["distance"], max_distance)
            cost = random.uniform(0,10) # 这个不重要

            data["links"].append({
                "src": u,
                "snk": v,
                "cost": cost,
                "distance": attr["distance"],
                "ots": ots,
                "osnr": osnr,
                "colors": self.generate_colors()
            })

        json_name = "new_example/gabriel_network_" + str(num_nodes) + "_" + str(seed) + ".json"

        # 写入 JSON 文件（格式化缩进便于阅读）
        with open(json_name, "w") as f:
            json.dump(data, f, indent=4)

        print("JSON 文件已生成：gabriel_network.json")

        self.json_to_csv(json_name)

        self.generate_services_to_csv(G)

        self.generate_relays_to_csv(G)






if __name__ == '__main__':
    num_of_nodes = [20]
    for i in range(len(num_of_nodes)):
        num_nodes = num_of_nodes[i]
        seed = 0 # 随机种子，保证可复现
        c_max = 968
        band = 8
        target_occupacy = 0.2 # 初始网络占用率
        num_service = 20 # 要产生的业务数量
        relay_ratio = 0.3 # 中继器的比例，一般在0.1~0.3之间，越高，服务成功率也会相应高一些
        file_name = "example_" + str(num_nodes)

        GG = Gabriel_Generator(num_nodes, seed, c_max, band, target_occupacy, file_name, num_service, relay_ratio)
        GG.main()
