# 测试图类能不能实现正常功能
import sys
import os

# 手动添加包路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..//src')))
from protein_network.graph import create_undirected_connected_graph
from protein_network.graph import create_graph_from_file

if __name__ == '__main__':
    m=create_undirected_connected_graph(10, 0.1)
    print(m)
    m.output_graph("test.txt")
    n=create_graph_from_file("test.txt")
    print(n)
