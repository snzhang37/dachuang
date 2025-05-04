import itertools
import json
import math
import random
import sys
import time
from networkx import NetworkXNoPath
from itertools import groupby
import numpy as np

import networkx as nx


nodes = 1000
# topo_num = 3
# 根据节点数和拓扑编号构造拓扑名称（路径：gabriel/<nodes>/<topo_num>）
# topo_name = 'new_example/gabriel_network_/%s/%s' % (nodes, topo_num)
topo_name = 'new_example/gabriel_network_%s' % (nodes)
# 从JSON文件中加载拓扑，并转换为NetworkX图对象
g = nx.node_link_graph(json.load(open('%s.json' % topo_name)))

pass