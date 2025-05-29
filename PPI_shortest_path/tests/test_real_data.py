# 用真实世界数据测试算法，这里用最快的dijkstra算法进行测试
import sys
import os

# 手动添加包路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..//src')))

# 现在可以正常导入
from protein_network.graph import Graph
from protein_network.graph import create_graph_from_file


if __name__ == '__main__':
    m=create_graph_from_file('PPI_shortest_path//data//4909.protein.links.v12.0.txt')
    path=m.dijkstra_shortest_path('4909.A0A099NQD3', '4909.Q9Y798')
    print(path)
