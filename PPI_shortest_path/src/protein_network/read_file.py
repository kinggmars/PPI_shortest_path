from graph import Graph

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
            graph.adj[protein1][protein2] = weight
            graph.adj[protein2][protein1] = weight
    return graph


'''测试程序
filename = "test.txt"  # 文件路径
graph = create_graph_from_file(filename)

# 打印图的邻接表
print("图的邻接表：")
for node, neighbors in graph.adj.items():
    print(f"{node}: {neighbors}")


'''
