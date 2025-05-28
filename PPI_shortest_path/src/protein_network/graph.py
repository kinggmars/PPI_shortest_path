# 使用邻接表的数据结构存储图，并实现图的简单操作
from collections import defaultdict

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
# tests/test_graph.py
# # 测试图类能不能实现正常功能
# import sys
# sys.path.append("..")
# from src.protein_network.graph import Graph
# if __name__ == '__main__':
#     # 用你的原始数据初始化图
#     initial_data = {
#         '1': {'2': 2, '4': 1},
#         '2': {'4': 3, '5': 11},
#         '3': {'1': 4, '6': 5},
#         '4': {'3': 2, '6': 8, '7': 4, '5': 2},
#         '5': {'7': 6},
#         '7': {'6': 1}
#     }

#     g = Graph(initial_data)

#     # 打印图的邻接表
#     print("--- 初始图结构 ---")
#     print(g)

#     # 操作示例
#     print("\n--- 操作测试 ---")
#     print("节点'1'的邻接节点:", list(g.get_neighbors('1')))  # [('2', 2), ('4', 1)]
#     print("边'4'->'5'的权重:", g.get_edge_weight('4', '5'))  # 2
#     print("是否存在边'7'->'6':", g.has_edge('7', '6'))      # True

#     # 添加新边
#     g.add_edge('6', '3', weight=7)
#     print("\n添加边 6->3 后:")
#     print(g.adj['6'])  # {'3': 7}

#     # 删除节点'5'
#     g.remove_node('5')
#     print("\n删除节点'5'后:")
#     print(g)

