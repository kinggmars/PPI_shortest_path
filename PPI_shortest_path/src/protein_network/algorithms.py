# 四个算法
from graph import Graph
import numpy as np

#Floyd-Warshall
def floyd_warshall(graph:Graph):
    n = len(graph.adj)
    # 深拷贝原始图，避免修改输入数据,以及利用graph中储存的邻接表生成方便该算法编写的邻接矩阵
    path_matrix=[[None]*n for i in range(n)]#每个点对之间生成一个路径矩阵
    dist=np.zeros((n,n),dtype=float)#初始化距离矩阵，距离全部设置为零，并在下一步更新距离，

    #邻接矩阵的初始化
    node_to_id={}#初始化两个节点编号的字典，方便形成邻接矩阵
    id_to_node={}
    i=0
    for node in graph.adj.keys():
        node_to_id[node]=i
        id_to_node[i]=node
        i+=1
    for node in graph.adj.keys():
        its_index=node_to_id[node]
        for index in range(n):
            if index==its_index:
                pass
            else:
                next_node=id_to_node[index]
                if next_node in graph.adj[node].keys():
                    weight=graph.adj[node][next_node]
                    dist[its_index][index]=weight
                    path_matrix[its_index][index]=[node,next_node]
                else:
                    dist[its_index][index]=float('inf')
    #完成邻接矩阵初始化，自身对自身距离为0，未连接

    # 三重循环更新所有节点对的最短路径
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] != float('inf') and dist[k][j] != float('inf'):
                    #如果路径更短则更新
                    if dist[i][j]<=dist[i][k]+dist[k][j]:
                        pass
                    else:
                        path_copy=path_matrix[i][k].copy()
                        path_copy.pop()
                        path_matrix[i][j]=path_copy+path_matrix[k][j]
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist,path_matrix





    



#Dijkstra
import heapq
# 算法的复杂度为O((m+n)logn)
def dijkstra_shortest_paths(graph, start_node):
    """
    使用 Dijkstra 算法计算从起点到所有节点的最短路径。
    返回两个字典：最短距离和前驱节点。
    """
    if start_node not in graph.adj:
        raise ValueError(f"Start node '{start_node}' not found in the graph.")
    
    # 初始化距离和前驱字典
    distances = {node: float('inf') for node in graph.adj}
    predecessors = {node: None for node in graph.adj}
    distances[start_node] = 0
    
    # 使用优先队列（最小堆）
    heap = [(0, start_node)]
    
    while heap:
        current_distance, u = heapq.heappop(heap)
        
        # 如果当前距离大于已知最短距离，跳过
        if current_distance > distances[u]:
            continue
        
        # 遍历邻接节点
        for v, weight in graph.get_neighbors(u):
            distance_through_u = current_distance + weight
            # 如果找到更短的路径，更新
            if distance_through_u < distances[v]:
                distances[v] = distance_through_u
                predecessors[v] = u
                heapq.heappush(heap, (distance_through_u, v))
    
    return distances, predecessors

def dijkstra_shortest_path(graph, start_node, end_node):
    """
    返回从起点到终点的最短路径及其总权重。
    如果路径不存在，返回 (None, inf)。
    """
    if start_node not in graph.adj or end_node not in graph.adj:
        raise ValueError("Start or end node not in graph.")
    
    distances, predecessors = dijkstra_shortest_paths(graph, start_node)
    if distances[end_node] == float('inf'):
        return None, float('inf')
    
    # 反向构建路径
    path = []
    current = end_node
    while current is not None:
        path.append(current)
        current = predecessors[current]
    path.reverse()
    
    return path, distances[end_node]

def dijkstra_export_all_paths(graph, filename):
    """
    导出所有节点对的最短路径到文件，格式为：
    node1 node2 total_weight path
    例如：A B 5 A->B->C
    """
    with open(filename, 'w') as f:
        nodes = list(graph.adj.keys())  # 获取所有节点
        f.write("node1 node2 total_weight path\n")
        for u in nodes:
            try:
                # 计算以 u 为起点的最短路径
                distances, predecessors = dijkstra_shortest_paths(graph, u)
            except ValueError:
                continue  
            
            for v in nodes:
                if u == v:
                    continue  # 跳过自身
                if distances[v] == float('inf'):
                    continue  # 不可达的路径不写入
                
                # 重建路径
                path = []
                current = v
                while current is not None:
                    path.append(current)
                    current = predecessors[current]
                path.reverse()  # 反转得到正确顺序
                
                # 将路径转换为字符串（如 A->B->C）
                path_str = "->".join(path)
                # 写入文件：u v weight path
                f.write(f"{u} {v} {distances[v]} {path_str}\n")



#Bellman-Ford

# algorithms.py


def bellman_ford_shortest_paths(graph, start):
    """
    Bellman-Ford算法实现，输出格式为：起点 终点 总权重 路径
    
    参数：
    graph (Graph): 图实例
    start (str): 起点节点
    
    返回：
    list: 每个元素为格式化的路径字符串

    示例：
    graph:
        'A', 'B', 1
        'A', 'C', 4
        'B', 'C', 2
        'B', 'D', 5
        'C', 'D', 1
    返回：
        ['A A 0 A', 'A B 1 A->B', 'A C 3 A->B->C', 'A D 4 A->B->C->D']
    """
    # 检查起始节点是否存在
    if start not in graph.adj:
        raise KeyError(f"起始节点 '{start}' 不存在于图中")
    
    # 初始化数据
    nodes = graph.adj.keys()
    distances = {n: float('inf') for n in nodes}
    predecessors = {n: None for n in nodes}
    distances[start] = 0
    
    # 松弛操作
    for _ in range(len(nodes) - 1):
        for u in graph.adj:
            for v, weight in graph.get_neighbors(u):
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    predecessors[v] = u
    
    # 生成路径输出
    results = []
    for node in nodes:        
        path = []
        current = node
        # 回溯路径
        while current is not None:
            path.append(current)
            current = predecessors[current]
        path.reverse()
        
        if not path or path[0] != start:
            # 不可达的情况
            results.append(f"{start} {node} inf")
        else:
            total_weight = distances[node]
            path_str = "->".join(path)
            results.append(f"{start} {node} {total_weight} {path_str}")
    
    return results

def bellmanford_export_all_paths(results, filename):
    """
    将Bellman-Ford算法的结果写入文件
    
    参数：
    results (list): Bellman-Ford算法返回的路径结果列表
    filename (str): 输出文件名
    
    返回：
    bool: 写入成功返回True，失败返回False

    调用示例：
    # 在调用Bellman-Ford后使用
    # results = bellman_ford(graph, 'A')
    # write_bf_results(results, "output.txt")
    输出示例：
        node1 node2 total_weight path
        A A 0 A
        A B 1 A->B
        A C 3 A->B->C
        A D 4 A->B->C->D
    理论上所有函数的返回值都可以通过此函数来写入文件
    
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("node1 node2 total_weight path\n")
            for line in results:
                f.write(f"{line}\n")
        return True
    except IOError as e:
        print(f"文件写入失败: {str(e)}")
        return False
    
def bellmanford_shortest_path(graph, start_node, end_node):
    """
    根据Bellman-Ford算法获取从起始节点到目标节点的最短路径结果。

    参数:
        graph (Graph): 图实例。
        start_node (str): 起始节点。
        end_node (str): 目标节点。

    返回:
        list: 包含单个元素的列表，格式为'node1 node2 total_weight path'，若不可达则总权重为inf。
    """
    # 检查目标节点是否存在于图中
    if end_node not in graph.adj:
        return [f"{start_node} {end_node} inf"]
    
    # 调用Bellman-Ford算法获取所有结果
    try:
        results = bellman_ford(graph, start_node)
    except KeyError as e:
        raise e
    
    # 在结果中查找目标节点
    for line in results:
        parts = line.split()
        if parts[1] == end_node:
            return [line]
    
    # 理论上不会执行到此处
    return [f"{start_node} {end_node} inf"]
#Johnson
