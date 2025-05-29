# 使用邻接表的数据结构存储图，并实现图的简单操作
from collections import defaultdict
import heapq
import numpy as np
import random
import string
class Graph:
    def __init__(self, initial_graph=None):
        # 初始化图，可选参数为初始图数据（字典嵌套字典格式）
        self.adj = defaultdict(dict)
        if initial_graph: # 拷贝
            for node, neighbors in initial_graph.items():
                self.adj[node].update(neighbors)

    def add_node(self, node):
        # 添加节点（如果节点已存在则无操作）
        if node not in self.adj:
            self.adj[node] = {}

    def add_edge(self, u, v, weight=1):
        # 添加边 u->v 并设置权重（默认权重为1）
        self.add_node(u)
        self.add_node(v)
        self.adj[u][v] = weight

    def get_neighbors(self, node):
        """获取节点的所有邻接节点及其权重"""
        return self.adj[node].items() if node in self.adj else []

    def get_edge_weight(self, u, v):
        """获取边 u->v 的权重（若边不存在返回None）"""
        return self.adj[u].get(v, None) if u in self.adj else None

    def has_edge(self, u, v):
        """检查是否存在边 u->v"""
        return v in self.adj[u] if u in self.adj else False

    def remove_edge(self, u, v):
        """删除边 u->v"""
        if u in self.adj and v in self.adj[u]:
            del self.adj[u][v]

    def remove_node(self, node):
        """删除节点及其所有关联边"""
        if node in self.adj:
            del self.adj[node]
        # 删除其他节点指向该节点的边
        for n in self.adj:
            if node in self.adj[n]:
                del self.adj[n][node]

    def __str__(self):
        """可视化图的邻接表结构"""
        return "\n".join(
            f"{node}: {neighbors}"
            for node, neighbors in self.adj.items()
        )
    
    def output_graph(self, filename):
        """将图输出为文件"""
        with open(filename, 'w') as f:
            f.write("protein1 protein2 combined_score\n")
            for node, neighbors in self.adj.items():
                for neighbor, weight in neighbors.items():
                    f.write(f"{node} {neighbor} {1000-weight}\n")

    def floyd_warshall(self):
        n = len(self.adj)
        # 深拷贝原始图，避免修改输入数据,以及利用graph中储存的邻接表生成方便该算法编写的邻接矩阵
        path_matrix=[[None]*n for i in range(n)]#每个点对之间生成一个路径矩阵
        dist=np.zeros((n,n),dtype=float)#初始化距离矩阵，距离全部设置为零，并在下一步更新距离，

        #邻接矩阵的初始化
        node_to_id={}#初始化两个节点编号的字典，方便形成邻接矩阵
        id_to_node={}
        i=0
        for node in self.adj.keys():
            node_to_id[node]=i
            id_to_node[i]=node
            i+=1
        for node in self.adj.keys():
            its_index=node_to_id[node]
            for index in range(n):
                if index==its_index:
                    pass
                else:
                    next_node=id_to_node[index]
                    if next_node in self.adj[node].keys():
                        weight=self.adj[node][next_node]
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

    def floyd_warshall_export(self,filename):
        dist_matrix,path_matrix,id_to_node_dict=self.floyd_warshall()
        n=len(self)
        with open(filename,'w') as file:
            file.write("node1 node2 total_weight path\n")
            for row in range(n):
                for column in range(n):
                    distance=dist_matrix[row][column]
                    if distance==float('inf'):#未联通则跳过
                        pass
                    else:
                        start_node=id_to_node_dict[row]
                        end_node=id_to_node_dict[column]
                        file.write(f'{start_node} {end_node} {distance} ')
                        index_path_list=path_matrix[row][column]#以索引表示的路径
                        path_list=[id_to_node_dict[index] for index in index_path_list]
                        path='->'.join(path_list)
                        file.write(f'{path}\n')

    #Dijkstra
    import heapq
    # 算法的复杂度为O((m+n)logn)
    def dijkstra_shortest_paths(self, start_node):
        if start_node not in self.adj:
            raise ValueError(f"Start node '{start_node}' not found in the graph.")

        # 初始化距离和前驱字典
        distances = {node: float('inf') for node in self.adj}
        predecessors = {node: None for node in self.adj}
        distances[start_node] = 0

        # 使用优先队列（最小堆）
        heap = [(0, start_node)]

        while heap:
            current_distance, u = heapq.heappop(heap)

            # 如果当前距离大于已知最短距离，跳过
            if current_distance > distances[u]:
                continue
            
            # 遍历邻接节点
            for v, weight in self.get_neighbors(u):
                distance_through_u = current_distance + weight
                # 如果找到更短的路径，更新
                if distance_through_u < distances[v]:
                    distances[v] = distance_through_u
                    predecessors[v] = u
                    heapq.heappush(heap, (distance_through_u, v))

        # 生成路径输出
        results = []
        for node in self.adj:
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

    def dijkstra_shortest_path(self, start_node, end_node):
        # 检查节点是否存在
        if start_node not in self.adj or end_node not in self.adj:
            return f"{start_node} {end_node} inf"

        # 使用dijkstra_shortest_paths函数计算结果
        try:
            results = self.dijkstra_shortest_paths(start_node)
        except ValueError:
            return f"{start_node} {end_node} inf"

        # 在结果中查找目标节点
        for line in results:
            parts = line.split()
            if parts[1] == end_node:
                return line

        return f"{start_node} {end_node} inf"

    def dijkstra_export_all_paths(self, filename):
        try:
            with open(filename, 'w') as f:
                f.write("node1 node2 total_weight path\n")
                for start_node in self.adj:
                    try:
                        # 计算以 start_node 为起点的最短路径
                        paths = self.dijkstra_shortest_paths(self, start_node)
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

    def bellmanford_shortest_paths(self, start):
        
        # 检查起始节点是否存在
        if start not in self.adj:
            raise KeyError(f"起始节点 '{start}' 不存在于图中")

        # 初始化数据
        nodes = self.adj.keys()
        distances = {n: float('inf') for n in nodes}
        predecessors = {n: None for n in nodes}
        distances[start] = 0

        # 获取所有边
        edges = []
        for u in self.adj:
            for v, weight in self.adj[u].items():
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

    def bellmanford_export_all_paths(results, filename):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("node1 node2 total_weight path\n")
                for line in results:
                    f.write(f"{line}\n")
            return True
        except IOError as e:
            print(f"文件写入失败: {str(e)}")
            return False

    def bellmanford_shortest_path(self, start_node, end_node):
        # 检查目标节点是否存在于图中
        if end_node not in self.adj:
            return f"{start_node} {end_node} inf"

        # 调用Bellman-Ford算法获取所有结果
        try:
            results = self.bellmanford_shortest_paths(start_node)
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
    def johnson_shortest_paths(self, start_node):
        # 步骤1: 添加虚拟节点并运行Bellman-Ford
        virtual_node = "__JOHNSON_VIRTUAL__"
        temp_graph = Graph(initial_graph=self.adj)

        # 确保虚拟节点名称唯一
        while virtual_node in temp_graph.adj:
            virtual_node += "_"

        # 添加虚拟节点到所有其他节点的边（权重为0）
        for node in self.adj:
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
        for u in self.adj:
            for v, weight in self.adj[u].items():
                new_weight = weight + h[u] - h[v]
                reweighted_graph.add_edge(u, v, new_weight)

        # 步骤3: 在重新赋权图上运行Dijkstra
        distances, predecessors = reweighted_graph.dijkstra_shortest_paths(start_node)

        # 生成路径输出
        results = []
        for node in self.adj:
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

    def johnson_shortest_path(self, start_node, end_node):
        # 检查节点是否存在
        if start_node not in self.adj or end_node not in self.adj:
            return f"{start_node} {end_node} inf"

        # 使用johnson_shortest_paths函数计算结果
        try:
            results = self.johnson_shortest_paths(start_node)
        except RuntimeError as e:
            raise e

        # 在结果中查找目标节点
        for line in results:
            parts = line.split()
            if parts[1] == end_node:
                return line

        return f"{start_node} {end_node} inf"

    def johnson_export_all_paths(self, filename):
        try:
            with open(filename, 'w') as f:
                f.write("node1 node2 total_weight path\n")
                for start_node in self.adj:
                    try:
                        # 计算以 start_node 为起点的最短路径
                        paths = self.johnson_shortest_paths(start_node)
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
    dist, path_matrix, id_to_node = floyd_warshall(graph)
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
import heapq
# 算法的复杂度为O((m+n)logn)
def dijkstra_shortest_paths(graph, start_node):
    """
    使用Dijkstra算法计算从起点到所有节点的最短路径。
    返回格式：列表，每个元素为格式化的路径字符串 "起点 终点 总权重 路径"
    
    参数:
        graph (Graph): 图实例
        start_node (str): 起始节点
        
    返回:
        list: 包含所有路径的字符串列表
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

def dijkstra_shortest_path(graph, start_node, end_node):
    """
    使用Dijkstra算法计算从起点到终点的最短路径
    
    参数:
        graph (Graph): 图实例
        start_node (str): 起始节点
        end_node (str): 目标节点
        
    返回:
        str: 格式化的路径字符串 "起点 终点 总权重 路径"
    """
    # 检查节点是否存在
    if start_node not in graph.adj or end_node not in graph.adj:
        return f"{start_node} {end_node} inf"
    
    # 使用dijkstra_shortest_paths函数计算结果
    try:
        results = dijkstra_shortest_paths(graph, start_node)
    except ValueError:
        return f"{start_node} {end_node} inf"
    
    # 在结果中查找目标节点
    for line in results:
        parts = line.split()
        if parts[1] == end_node:
            return line
    
    return f"{start_node} {end_node} inf"

def dijkstra_export_all_paths(graph, filename):
    """
    导出所有节点对的最短路径到文件
    
    参数:
        graph (Graph): 图实例
        filename (str): 输出文件名
    
    返回:
        bool: 操作是否成功
    """
    try:
        with open(filename, 'w') as f:
            f.write("node1 node2 total_weight path\n")
            for start_node in graph.adj:
                try:
                    # 计算以 start_node 为起点的最短路径
                    paths = dijkstra_shortest_paths(graph, start_node)
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

# algorithms.py


def bellmanford_shortest_paths(graph, start):
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
        return f"{start_node} {end_node} inf"
    
    # 调用Bellman-Ford算法获取所有结果
    try:
        results = bellmanford_shortest_paths(graph, start_node)
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
    """
    使用Johnson算法计算从起点到所有节点的最短路径。
    返回格式与其他算法一致：列表，每个元素为格式化的路径字符串 "起点 终点 总权重 路径"
    
    参数:
        graph (Graph): 图实例
        start_node (str): 起始节点
        
    返回:
        list: 包含所有路径的字符串列表
    """
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
        h_results = bellmanford_shortest_paths(temp_graph, virtual_node)
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
    distances, predecessors = dijkstra_shortest_paths(reweighted_graph, start_node)
    
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
    """
    使用Johnson算法计算从起点到终点的最短路径
    
    参数:
        graph (Graph): 图实例
        start_node (str): 起始节点
        end_node (str): 目标节点
        
    返回:
        str: 格式化的路径字符串 "起点 终点 总权重 路径"
    """
    # 检查节点是否存在
    if start_node not in graph.adj or end_node not in graph.adj:
        return f"{start_node} {end_node} inf"
    
    # 使用johnson_shortest_paths函数计算结果
    try:
        results = johnson_shortest_paths(graph, start_node)
    except RuntimeError as e:
        raise e
    
    # 在结果中查找目标节点
    for line in results:
        parts = line.split()
        if parts[1] == end_node:
            return line
    
    return f"{start_node} {end_node} inf"

def johnson_export_all_paths(graph, filename):
    """
    使用Johnson算法导出所有节点对的最短路径到文件
    
    参数:
        graph (Graph): 图实例
        filename (str): 输出文件名
    
    返回:
        bool: 操作是否成功
    """
    try:
        with open(filename, 'w') as f:
            f.write("node1 node2 total_weight path\n")
            for start_node in graph.adj:
                try:
                    # 计算以 start_node 为起点的最短路径
                    paths = johnson_shortest_paths(graph, start_node)
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

def create_graph_from_file(filename):
    """
    从文件中读取数据并构建无向图。
    
    参数：
    filename (str): 包含蛋白质交互数据的文件路径
    
    返回：
    Graph: 构建的无向图实例
    """
    graph = Graph()
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            parts = line.split()
            if len(parts) != 3:
                continue  # 跳过格式不正确的行
            protein1, protein2, combined_score = parts
            try:
                weight = 1000 - int(combined_score)
            except ValueError:
                continue  # 如果转换失败，跳过该行
            # 添加无向边，两个方向都设置相同的权重
            graph.add_edge(protein1, protein2, weight)
            graph.add_edge(protein2, protein1, weight)
    return graph

    
    
