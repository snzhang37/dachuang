import csv
import networkx as nx

# 这个代码是从CSV文件中提取出我们需要的拓扑形式，与csv_out很类似

# The colors data is processed as f_slot, and n colors are merged into one band
# 将colors数据处理为频隙，n个color合并为一个波段
# 为了灵活产生例子
def process_colors(colors, n, c_max): # 其实这个c_max可以从文件里读取的
    N = int(c_max/n) # 频隙个数
    # f_slot = [0 for i in range(N)]
    f_slot = set() # 直接存可用频隙，而不是所有频隙
    segments = colors.split(':')
    while segments:
        s = segments.pop(0)
        if s == '':
            continue
        start, end = map(int, s.split('-'))
        for j in range(int(start/n), int(end/n)):
            # f_slot[j] = 1
            f_slot.add(j)

    return f_slot

# 注意！ band 是根据service文件中业务所需最小间隙数决定的

def create_topology(file_path, file_name, band, c_max, **kwargs):
    network = nx.MultiGraph()  # multigraph  # 多重无向图，运行平行边
    # network = nx.Graph() # 单边无向图

    network.graph['L'] = int(c_max/band) # 频隙数量

    # Read the node.csv file and add nodes to the graph
    # 读取节点文件，向图中添加节点
    with open(file_path+file_name+'.node.csv') as f1:
        Node = csv.DictReader(f1)
        node_list = []
        for n in Node:
            # relay:是否为中继节点；available relay num:该节点可以进行多少中继；available relay:存储具体的中继信息
            node_list.append((int(n['nodeId']), {'relay': False, 'available relay': [], 'degree':0}))# 聚类过程中使用，判断有无遗失的边

            # network.add_node(n['nodeId'], 'relay'=False)
        # network.add_nodes_from(node_list, relay=False, available_relay_num=0, available_relay=[])
        network.add_nodes_from(node_list)

    # Read the oms.csv file and add edges to the graph
    # 读取oms文件，向图中添加边
    # with open(file_path + file_name + '.oms.csv') as f2:
    with open(file_path + file_name + '.oms_processed.csv') as f2:
        Edge = csv.DictReader(f2)
        edge_list = []
        for e in Edge:
            if len(e['colors']) > 0: # 只添加存在有效频隙的边

                edge_list.append((int(e['src']), int(e['snk']), # 源节点，目的节点
                                  {'distance': float(e['distance']),# 本条链路的距离长度(km)
                                   'ots': float(e['ots']), # 本条链路经过的ots跳数 # ots:光传输段，另一个物理层，用于管理和监控光信号在物理光纤中的传输质量 # 基本都是1，表示光纤上的数据都被收集到
                                   'osnr': float(e['osnr']), # 本oms链路的osnr（光信噪比）值
                                   'f_slot': process_colors(e['colors'], n=band,c_max=c_max) # 本oms链路的频隙占用情况
                                   }))

        network.add_edges_from(edge_list)
        # print(edge_list)



    # Process the relay.csv file to add available relay attributes to the node
    # 处理relay文件，向节点添加可用中继属性
    with open(file_path + file_name + '.relay.csv') as f3:
        Relay = csv.DictReader(f3)
        for r in Relay:
            network.nodes[int(r['nodeId'])]['relay'] = True
            network.nodes[int(r['nodeId'])]['available relay'].append({'available': True,})
        # nx.draw(network, with_labels=True)
        # plt.show()
    return network

def process_service(file_path, file_name, band=24, c_max = 964):
    service_list = []
    with open(file_path+file_name+'.service.csv') as f1:
        Service = csv.DictReader(f1)
        for s in Service:
            service_list.append(
                {'Index': s['Index'], # 业务需求ID
                 'src': int(s['src']), # 业务需求起始点ID
                 'snk': int(s['snk']), # 业务需求终点ID
                 }
            )
    # print(service_list)
    return service_list

# 获取中继节点
def get_relay_node(G):
    relay_node = []
    for n in G.nodes():
        if G.nodes[n]['relay']:
            relay_node.append(n)
    return relay_node
