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
def bellman_ford(graph, start):
    """
    Bellman-Ford算法实现，计算从起点到所有其他节点的最短路径。
    
    参数：
    graph (Graph): 图实例
    start (str): 起点节点
    
    返回：
    dict: 从起点到每个节点的最短路径长度

    示例：
    graph:
    'A', 'B', 1
    'A', 'C', 4
    'B', 'C', 2
    'B', 'D', 5
    'C', 'D', 1
    调用：bellman_ford(graph, 'A')
    返回：{'A': 0, 'B': 1, 'C': 3, 'D': 4}
    """
    # 检查起始节点是否存在
    if start not in graph.adj:
        raise KeyError(f"起始节点 {start} 不存在于图中")
    # 初始化距离字典
    distances = {node: float('inf') for node in graph.adj}
    distances[start] = 0
    # 松弛操作
    for _ in range(len(graph.adj) - 1):
        for u in graph.adj:
            for v, weight in graph.get_neighbors(u):
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
    # 检测负权重环路
    for u in graph.adj:
        for v, weight in graph.get_neighbors(u):
            if distances[u] + weight < distances[v]:
                raise ValueError("图中存在负权重环路")
    return distances

#Johnson
