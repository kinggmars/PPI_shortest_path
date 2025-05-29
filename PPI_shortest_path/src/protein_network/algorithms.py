from graph import Graph
import heapq
import numpy as np
import random
import string
#Floyd-Warshall
def floyd_warshall(graph):
    # 创建节点映射，方便在矩阵运算后进行查找蛋白质名称
    nodes = list(graph.adj.keys())
    n = len(nodes)
    node_to_id = {node: i for i, node in enumerate(nodes)}
    id_to_node = {i: node for i, node in enumerate(nodes)}
    # 初始化距离矩阵和路径矩阵
    dist = [[float('inf')] * n for _ in range(n)]
    path_matrix = [[None] * n for _ in range(n)]
    # 设置对角线：自身到自身
    for i in range(n):
        dist[i][i] = 0
        path_matrix[i][i] = [nodes[i]]  # 存储路径
    # 直接相邻的边也需要更改距离与路径
    for u in graph.adj:
        u_id = node_to_id[u]
        for v, weight in graph.adj[u].items():
            v_id = node_to_id[v]
            dist[u_id][v_id] = weight
            path_matrix[u_id][v_id] = [u, v]  
    # 算法核心运算部分：三重循环更新最短路径
    for k in range(n):
        for i in range(n):
            if dist[i][k] == float('inf'):
                continue
            for j in range(n):
                if dist[k][j] == float('inf'):
                    continue
                new_dist = dist[i][k] + dist[k][j]
                if new_dist < dist[i][j]:
                    dist[i][j] = new_dist
                    # 正确合并路径：i->k + k->j (注意需要去掉重复的k节点)
                    path_matrix[i][j] = path_matrix[i][k] + path_matrix[k][j][1:]
    return dist, path_matrix, id_to_node
def floyd_warshall_export(graph, filename):
    dist, path_matrix, id_to_node = graph.floyd_warshall()
    n = len(dist)
    with open(filename, 'w') as f:
        f.write("node1 node2 total_weight path\n")
        for i in range(n):
            for j in range(n):
                if dist[i][j] == float('inf'):
                    continue  # 跳过不可达的节点对
                if i==j:
                    continue #自身无需再打印了
                
                start = id_to_node[i]
                end = id_to_node[j]
                weight = dist[i][j]
                path = "->".join(path_matrix[i][j])  # 路径已存储为节点名称
                f.write(f"{start} {end} {weight} {path}\n")
#Dijkstra
# 算法的复杂度为O((m+n)logn)
def dijkstra_shortest_paths(graph, start_node):
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
    # 生成路径输出
    results = []
    for node in graph.adj:
        if distances[node] == float('inf'):
            results.append(f"{start_node} {node} inf")
            continue
        # 重建路径
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = predecessors[current]
        path.reverse()
        path_str = "->".join(path)
        results.append(f"{start_node} {node} {distances[node]} {path_str}")
    return results

def dijkstra(graph, start_node):
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
    # 检查节点是否存在
    if start_node not in graph.adj or end_node not in graph.adj:
        return f"{start_node} {end_node} inf"
    # 使用dijkstra_shortest_paths函数计算结果
    try:
        results = graph.dijkstra_shortest_paths(start_node)
    except ValueError:
        return f"{start_node} {end_node} inf"
    # 在结果中查找目标节点
    for line in results:
        parts = line.split()
        if parts[1] == end_node:
            return line
    return f"{start_node} {end_node} inf"
def dijkstra_export_all_paths(graph, filename):
    try:
        with open(filename, 'w') as f:
            f.write("node1 node2 total_weight path\n")
            for start_node in graph.adj:
                try:
                    # 计算以 start_node 为起点的最短路径
                    paths = graph.dijkstra_shortest_paths(start_node)
                except ValueError:
                    continue
                
                # 写入结果
                for path_str in paths:
                    # 跳过自身路径（可选）
                    parts = path_str.split()
                    if parts[0] == parts[1]:
                        continue
                    f.write(f"{path_str}\n")
        return True
    except IOError as e:
        print(f"文件写入失败: {str(e)}")
        return False
#Bellman-Ford
def bellmanford_shortest_paths(graph, start):
    
    # 检查起始节点是否存在
    if start not in graph.adj:
        raise KeyError(f"起始节点 '{start}' 不存在于图中")
    # 初始化数据
    nodes = graph.adj.keys()
    distances = {n: float('inf') for n in nodes}
    predecessors = {n: None for n in nodes}
    distances[start] = 0
    # 获取所有边
    edges = []
    for u in graph.adj:
        for v, weight in graph.adj[u].items():
            edges.append((u, v, weight))
    # 松弛操作
    for _ in range(len(nodes) - 1):
        updated = False
        for u, v, w in edges:
            if distances[u] + w < distances[v]:
                distances[v] = distances[u] + w
                predecessors[v] = u
                updated = True
        if not updated:  # 提前终止优化
            break
        
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
def bellmanford_export_all_paths(graph, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for node in graph.adj:
            results = graph.bellmanford_shortest_paths(node)
            try:
                f.write("node1 node2 total_weight path\n")
                for line in results:
                    f.write(f"{line}\n")
            except IOError as e:
                print(f"文件写入失败: {str(e)}")
                return False
    return True
def bellmanford_shortest_path(graph, start_node, end_node):
    # 检查目标节点是否存在于图中
    if end_node not in graph.adj:
        return f"{start_node} {end_node} inf"
    # 调用Bellman-Ford算法获取所有结果
    try:
        results = graph.bellmanford_shortest_paths(start_node)
    except KeyError as e:
        raise e
    # 在结果中查找目标节点
    for line in results:
        parts = line.split()
        if parts[1] == end_node:
            return line
    # 理论上不会执行到此处
    return f"{start_node} {end_node} inf"

#Johnson
def johnson_shortest_paths(graph, start_node):
    # 步骤1: 添加虚拟节点并运行Bellman-Ford
    virtual_node = "__JOHNSON_VIRTUAL__"
    temp_graph = Graph(initial_graph=graph.adj)
    # 确保虚拟节点名称唯一
    while virtual_node in temp_graph.adj:
        virtual_node += "_"
    # 添加虚拟节点到所有其他节点的边（权重为0）
    for node in graph.adj:
        temp_graph.add_edge(virtual_node, node, 0)
    # 运行Bellman-Ford获取势能函数h
    try:
        h_results = temp_graph.bellmanford_shortest_paths(virtual_node)
    except (KeyError, ValueError) as e:
        raise RuntimeError(f"Johnson算法初始化失败: {str(e)}")
    # 解析结果，得到h值（重新赋权函数）
    h = {}
    for line in h_results:
        parts = line.split()
        node = parts[1]
        if parts[2] == 'inf':
            raise RuntimeError(f"节点 {node} 从虚拟节点不可达")
        h[node] = float(parts[2])
    # 步骤2: 创建重新赋权后的图
    reweighted_graph = Graph()
    for u in graph.adj:
        for v, weight in graph.adj[u].items():
            new_weight = weight + h[u] - h[v]
            reweighted_graph.add_edge(u, v, new_weight)
    # 步骤3: 在重新赋权图上运行Dijkstra
    distances, predecessors = reweighted_graph.dijkstra(start_node)
    # 生成路径输出
    results = []
    for node in graph.adj:
        if distances[node] == float('inf'):
            results.append(f"{start_node} {node} inf")
            continue
        # 重建路径
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = predecessors[current]
        path.reverse()
        # 还原原始距离: d(u, v) = d'(u, v) - h(u) + h(v)
        original_dist = distances[node] - h[start_node] + h[node]
        path_str = "->".join(path)
        results.append(f"{start_node} {node} {original_dist} {path_str}")
    return results
def johnson_shortest_path(graph, start_node, end_node):
    # 检查节点是否存在
    if start_node not in graph.adj or end_node not in graph.adj:
        return f"{start_node} {end_node} inf"
    # 使用johnson_shortest_paths函数计算结果
    try:
        results = graph.johnson_shortest_paths(start_node)
    except RuntimeError as e:
        raise e
    # 在结果中查找目标节点
    for line in results:
        parts = line.split()
        if parts[1] == end_node:
            return line
    return f"{start_node} {end_node} inf"
def johnson_export_all_paths(graph, filename):
    try:
        with open(filename, 'w') as f:
            f.write("node1 node2 total_weight path\n")
            for start_node in graph.adj:
                try:
                    # 计算以 start_node 为起点的最短路径
                    paths = graph.johnson_shortest_paths(start_node)
                except RuntimeError as e:
                    f.write(f"# Error for source {start_node}: {str(e)}\n")
                    continue
                
                # 写入结果
                for path_str in paths:
                    # 跳过自身路径（可选）
                    parts = path_str.split()
                    if parts[0] == parts[1]:
                        continue
                    f.write(f"{path_str}\n")
        return True
    except IOError as e:
        print(f"文件写入失败: {str(e)}")
        return False
