#csv_write
import csv
import math
import random
import networkx as nx
from urllib3.filepost import writer
import numpy as np
import matplotlib.pyplot as plt

# CSV文件路径
ex='./new_example/example4.'
file_node =ex+ 'node.csv'
file_edge =ex+ 'oms.csv'
file_relay=ex+ 'relay.csv'
file_service=ex+ 'service.csv'

# 目标资源占用率（0~1），例如 0.3 表示 30% 频谱被占用
target_occupancy = 0.5

# 生成随机节点数
# N = random.randint(1000, 1500)
# N = random.randint(100, 200)

N = 100
N = 1000
# N = 500

# 生成随机节点ID
node_ids = random.sample(range(10, 5001), N)
nodes = [f"NODE,{node_id}" for node_id in node_ids]

# 写入node.csv文件
with open(file_node, mode='w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["NODE","nodeId"])
    for node in nodes:
        csvwriter.writerow([node.split(',')[0], node.split(',')[1]])  # 写入节点

# edge
repeats = {node_id: random.randint(1, 6) for node_id in node_ids}# 每个 nodeId 重复次数范围 (2~16)
count = 0
for repeat,repeat_value in repeats.items():
    count += repeat_value
ave = count / len(repeats)

# repeats = {node_id: 1 for node_id in node_ids}# 每个 nodeId 重复次数范围 1
edges = []
# print(repeats)
src_list = []
for node_id in sorted(repeats.keys()):  # 按 node_id 排序
    src_list.extend([node_id] * repeats[node_id])
snk_list = src_list[:]
random.shuffle(snk_list)
for i in range(len(src_list)):
    while snk_list[i] == src_list[i]:
        random.shuffle(snk_list)      #确保每条链路节点与源节点不同
# print(src_list)
# print(snk_list)
edges = list(zip(src_list, snk_list))

M = len(edges) # 边数量
inoms_ids = random.sample(range(M + 51), M)
oms_ids = inoms_ids[:]
random.shuffle(inoms_ids)
remote_oms_ids = inoms_ids[:]

# cost列表
cost_values = [20001] * (M//10) + [1] * 2
# 随机生成其他cost值,范围为21001到464007
other_costs = random.sample(range(21001, 464008, 1000), 190)
for cost in other_costs:
    cost_values.extend([cost] * random.randint(1, 20))  # 每个cost随机出现1到20次
# 确保 cost_values 的长度与 edges 一致
if len(cost_values) < M:
    cost_values.extend(random.choices(other_costs, k=M - len(cost_values)))
elif len(cost_values) > M:
    cost_values = cost_values[:M]
random.shuffle(cost_values)# 打乱 cost_values 列表

# distance列表
# 定义距离区间，格式为 (下界, 上界)
bins = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 300), (300, 350), (350, 400)]
# 对应每个区间的概率（需归一化，总和为1）
probabilities = [0.2659, 0.0780, 0.0815, 0.0886, 0.2198, 0.1312, 0.0673, 0.0638]
all_value = sum(probabilities)
probabilities = [item / all_value for item in probabilities]

# 利用 np.random.choice 根据给定概率选择每个边所属的区间的索引
chosen_bins = np.random.choice(len(bins), size=M, p=probabilities)

# 对每个选定的区间，在区间内均匀采样生成距离
distances = np.array([np.random.uniform(bins[i][0], bins[i][1]) for i in chosen_bins])
distance_values = list(distances)

# distance_values = [100] * M# 均匀

# ots列表
ots_choices = []
for distance in distance_values:
    if distance <= 100:
        ots_choices.append(1)
    elif distance <= 150:
        ots_choices.append(2)
    elif distance <= 161:
        ots = random.choice([2,3])
        ots_choices.append(ots)
    elif distance <= 196:
        ots_choices.append(3)
    elif distance <= 255:
        ots = random.choice([3, 4])
        ots_choices.append(ots)
    elif distance <= 320:
        ots = random.choice([4, 5])
        ots_choices.append(ots)
    elif distance <= 340:
        ots_choices.append(5)
    elif distance <= 374:
        ots = random.choice([5, 6])
        ots_choices.append(ots)
    elif distance <= 400:
        ots_choices.append(6)

# ots_choices = [1] * M # 均匀

# 创建osnr列表
osnr_choices = []
for distance in distance_values:
    min_value = 0.79809526e-05 * distance - 1.57785657e-04
    max_value = 2.58138093e-05 * distance + 2.47357677e-04
    osnr = random.uniform(min_value, max_value)
    osnr = max(1e-05, osnr)
    osnr_choices.append(osnr)

# osnr_choices = [0.000776] * M # 均匀


# 频谱范围[0,960]， L=120为频隙（波）数量
# rsop:目标资源占用率
def generate_colors(target_occupancy=0.5):
    c_max = 964
    # band = 24
    band = 8

    total_slots = math.floor(c_max/band)
    ranges = [f"{i}-{i + band}" for i in range(0, c_max, band)]

    used_slots = int(total_slots * (1-target_occupancy))  # 计算需要占用的频谱数

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

oms_data = []
for idx, (src, snk) in enumerate(edges):
    oms_data.append([
        "OMS",  # OMS
        oms_ids[idx],  # omsId
        remote_oms_ids[idx],  # remoteOmsId
        src,  # src
        snk,  # snk
        cost_values[idx],
        distance_values[idx],
        ots_choices[idx],
        osnr_choices[idx],
        6250,  # slice 固定值
        generate_colors(target_occupancy)
    ])
# 写入oms.csv文件
with open(file_edge, mode='w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["OMS", "omsId", "remoteOmsId", "src", "snk", "cost", "distance", "ots", "osnr", "slice", "colors"])
    csvwriter.writerows(oms_data)

#relay
relay_data = []
select_node_ids = random.sample(node_ids, random.randint(N // 10,N// 3)) #随机选取中继节点
select_node_ids.sort()
#每个中继节点上有偶数个中继器，且少量中继器出现概率大
even_counts = list(range(2, 200, 2))
weights = []
for count in even_counts:
    if 2 <= count <= 20:
        weights.append(40)  # 权重为 40，高概率
    elif 20< count <= 100:
        weights.append(5)  # 低概率
    else:
        weights.append(1)
relay_repeats = {
    select_node_id: random.choices(even_counts, weights=weights)[0] for select_node_id in select_node_ids
}
relay_node_ids = []
for select_node_id in select_node_ids:
    repeat_count = relay_repeats[select_node_id]
    relay_node_ids.extend([select_node_id] * repeat_count)
# print(select_node_ids)
# print(relay_node_ids)
R = len(relay_node_ids)
inrelay_ids = random.sample(range(10,10*R), R)
relay_ids = inrelay_ids[:]
random.shuffle(inrelay_ids)
relatedrelay_ids = inrelay_ids[:]

inlocal_ids = random.sample(range(10,2*R),R//50)
# 确保 inlocal_ids 的长度与 R 一致
if len(inlocal_ids) < R:
    inlocal_ids.extend(random.choices(inlocal_ids, k=R - len(inlocal_ids)))
elif len(inlocal_ids) > R:
    inlocal_ids = inlocal_ids[:R]
local_ids = inlocal_ids[:]
random.shuffle(inlocal_ids)
relatedlocal_ids = inlocal_ids[:]

for i,relay_node_id in enumerate(relay_node_ids):
    relay_data.append([
        "RELAY",
        relay_ids[i],  # relayId
        relatedrelay_ids[i],  # relatedRelayId
        relay_node_id, #中继节点的nodeId
        local_ids[i],# localId
        relatedlocal_ids[i],# relatedLocalId
        generate_colors() #dimColors
    ])

# 写入relay.csv文件
with open(file_relay, mode='w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["RELAY", "relayId", "relatedRelayId", "nodeId", "localId", "relatedLocalId", "dimColors"])
    csvwriter.writerows(relay_data)

# service
ser_repeats = {node_id: random.randint(1, 120) for node_id in node_ids}# 每个 nodeId 重复次数范围 (2~16)
ser_edges = []
for s_src in ser_repeats.keys():
    s_snk_list = [node for node in ser_repeats.keys() if node != s_src]  # 目的节点不能是源节点
    for _ in range(ser_repeats[s_src]):  # 根据 s_src 的重复次数生成边
        s_snk = random.choice(s_snk_list)  # 随机选择一个 snk
        ser_edges.append((s_src, s_snk))
S = len(ser_edges) # 服务请求数量
source_otu = random.sample(range(1000,1000+10*S),S)
target_otu = random.sample(range(1001,999+10*S),S)
service_data = []
for idx, (s_src, s_snk) in enumerate(ser_edges):
    service_data.append([
        idx,  # Index
        s_src,  # src
        s_snk,  # snk
        source_otu[idx],#sourceOtu
        target_otu[idx],#targetOtu
        8,  # m_width 固定值
        # 8,  # m_width 固定值
        0,  # bandType 固定值
        generate_colors(), #sourceDimColors
        generate_colors() #targetDimColors
    ])

# 写入service.csv文件
with open(file_service, mode='w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Index", "src", "snk", "sourceOtu", "targetOtu", "m_width", "bandType", "sourceDimColors", "targetDimColors"])
    csvwriter.writerows(service_data)
# N节点数量 M链路边数量 R中继器数量 S业务需求数量
print(N,M,R,S)