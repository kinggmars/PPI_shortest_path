# 比较算法时间复杂度
import sys
import os
import matplotlib.pyplot as plt
import time
# 手动添加包路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..//src')))

# 现在可以正常导入
from protein_network.graph import Graph
from protein_network.graph import create_undirected_connected_graph

node_counts = list(range(100,301 , 100))
sparse_floyd = []
sparse_dijkstra = []
sparse_bellman_ford = []
sparse_johnson = []
dense_floyd = []
dense_dijkstra = []
dense_bellman_ford = []
dense_johnson = []

for n in node_counts:
    # 稀疏图测试
    m = create_undirected_connected_graph(n,0.2)
    # floyd
    start = time.perf_counter()
    m.floyd_warshall_export('floyd.txt')
    sparse_floyd.append(time.perf_counter() - start)
    # dijkstra
    start = time.perf_counter()
    m.dijkstra_export_all_paths('dijkstra.txt')
    sparse_dijkstra.append(time.perf_counter() - start)
    # bellman_ford
    start = time.perf_counter()
    m.bellmanford_export_all_paths('bellman_ford.txt')
    sparse_bellman_ford.append(time.perf_counter() - start)
    # johnson
    start = time.perf_counter()
    m.johnson_export_all_paths('johnson.txt')
    sparse_johnson.append(time.perf_counter() - start)
    
    # 稠密图测试
    m = create_undirected_connected_graph(n,0.8)
    # floyd
    start = time.perf_counter()
    m.floyd_warshall_export('floyd.txt')
    dense_floyd.append(time.perf_counter() - start)
    # dijkstra
    start = time.perf_counter()
    m.dijkstra_export_all_paths('dijkstra.txt')
    dense_dijkstra.append(time.perf_counter() - start)
    # bellman_ford
    start = time.perf_counter()
    m.bellmanford_export_all_paths('bellman_ford.txt')
    dense_bellman_ford.append(time.perf_counter() - start)
    print(n)
    # johnson
    start = time.perf_counter()
    m.johnson_export_all_paths('johnson.txt')
    dense_johnson.append(time.perf_counter() - start)
# 绘图
plt.figure(figsize=(15, 6))

# 稀疏图
plt.subplot(1, 2, 1)
plt.plot(node_counts, sparse_floyd, label='Floyd-Warshall', marker='o',color='blue')
plt.plot(node_counts, sparse_dijkstra, label='Dijkstra', marker='x',color='red')
plt.plot(node_counts, sparse_bellman_ford, label='Bellman-Ford', marker='^',color='green')
plt.plot(node_counts, sparse_johnson, label='Johnson', marker='s',color='purple')
plt.xlabel('Number of Nodes')
plt.ylabel('Time (s)')
plt.title('Sparse Graph Performance')
plt.legend()

# 稠密图
plt.subplot(1, 2, 2)
plt.plot(node_counts, dense_floyd, label='Floyd-Warshall', marker='o',color='blue')
plt.plot(node_counts, dense_dijkstra, label='Dijkstra', marker='x',color='red')
plt.plot(node_counts, dense_bellman_ford, label='Bellman-Ford', marker='^',color='green')
plt.plot(node_counts, dense_johnson, label='Johnson', marker='s',color='purple')
plt.xlabel('Number of Nodes')
plt.ylabel('Time (s)')
plt.title('Dense Graph Performance')

plt.tight_layout()
plt.show()
