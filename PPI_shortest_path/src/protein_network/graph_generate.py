# 生成一个蛋白质网络
from graph import Graph
import random
import string
def create_undirected_connected_graph(num_nodes=100,sparse=0.01):
    """创建随机稀疏无向连通图，节点数默认100，稀疏度默认为0.01，稀疏度即为边数比最大边数"""
    graph = Graph()
    
    # 生成节点名,并且确保不重复
    nodes = set()
    while len(nodes) < num_nodes:
        part1="0000."
        part2 = ''.join(random.choices(string.ascii_uppercase, k=6))
        part3 = ''.join(str(random.randint(0,9))for _ in range(4))
        node = part1+part2+part3
        nodes.add(node)
    nodes = list(nodes)
    random.shuffle(nodes)
    
    if num_nodes == 0:
        return graph
    
    # 构建生成树确保连通性
    connected = {nodes[0]}
    tree_edges = set()
    for i in range(1, num_nodes):
        u = nodes[i]
        v = random.choice(list(connected))
        # 存储无向边（有序元组）
        edge = tuple(sorted((u, v)))
        tree_edges.add(edge)
        connected.add(u)
    
    # 添加生成树边（双向）
    for u, v in tree_edges:
        weight = random.randint(0, 1000)
        graph.add_edge(u, v, weight)
        graph.add_edge(v, u, weight)
    
    # 添加额外边
    extra_edges = set()
    possible_pairs = [
        tuple(sorted((nodes[i], nodes[j])))
        for i in range(num_nodes)
        for j in range(i+1, num_nodes)
    ]
    non_tree_pairs = [pair for pair in possible_pairs if pair not in tree_edges]
    k = int(num_nodes*(num_nodes-1)//2*sparse-num_nodes+1)# 计算需要额外添加的边数
    if k > 0:
        for pair in random.sample(non_tree_pairs, k):# 从没有加入生成树的边随机选k个
            weight = random.randint(0, 1000)
            graph.add_edge(pair[0], pair[1], weight)
            graph.add_edge(pair[1], pair[0], weight)

    return graph

# m=create_undirected_connected_graph(10,1)
# print(m)
    
    


