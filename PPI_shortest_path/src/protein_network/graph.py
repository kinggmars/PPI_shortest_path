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

