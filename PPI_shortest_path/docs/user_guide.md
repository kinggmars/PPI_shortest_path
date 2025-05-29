# 🧬 蛋白质网络图分析工具用户手册

## 目录

[toc]

## 1. 图数据结构（Graph类）



### 功能描述
实现基于邻接表的图数据结构，支持节点和边的增删改查操作,支持有权边。

### 核心方法

```python
# 创建图实例
graph = Graph()  # 空图
graph = Graph(initial_data)  # 用初始数据创建图

# 添加节点
graph.add_node("P12345")

# 添加边（带权重）
graph.add_edge("P12345", "Q98765", weight=850)

# 获取邻接节点
neighbors = graph.get_neighbors("P12345")

# 获取边权重
weight = graph.get_edge_weight("P12345", "Q98765")

# 检查边是否存在
exists = graph.has_edge("P12345", "Q98765")

# 删除边
graph.remove_edge("P12345", "Q98765")

# 删除节点
graph.remove_node("P12345")

# 可视化图结构
print(graph)  # 打印邻接表

# 导出图到文件
graph.output_graph("network.txt")
```


### 示例用法
```python
# 创建初始图
initial_data = {
    'P1': {'P2': 200, 'P3': 300},
    'P2': {'P3': 150, 'P4': 400},
    'P3': {'P4': 250}
}
g = Graph(initial_data)

# 添加新节点和边
g.add_node("P5")
g.add_edge("P4", "P5", weight=350)

# 输出图结构
print("当前图结构:")
print(g)

# 导出到文件
g.output_graph("protein_network.txt")



```


---

## 2.最短路径算法
**ps**：包含在graph类中

### 1.Floyd-Warshall算法
**特点**：全源最短路径，适合稠密图，但是生物分子相关网络较为稀疏，较为浪费空间与时间
**时间复杂度**：O(n³)

**使用实例**
```python
# 计算所有节点对的最短路径，其中dist_matrix以矩阵形式记录两点之间距离，path_matrix以[a,b,c]为路径格式记录最短路径，id_to_node为一个将矩阵索引对应蛋白名的字典
dist_matrix, path_matrix, id_to_node = g.floyd_warshall

# 导出结果到文件
'''格式
输出示例：
        node1 node2 total_weight path
        A A 0 A
        A B 1 A->B
        A C 3 A->B->C
        A D 4 A->B->C->D
'''

g.floyd_warshall_export( "floyd_results.txt")
```

### 2.Dijkstra算法
**特点**：单源最短路径，非负权重
**时间复杂度**：O((m+n)logn)
**使用实例**
```python
# 计算从指定起点到所有节点的最短路径
paths = g.dijkstra_shortest_paths( "P12345")

# 计算指定起点到终点的最短路径
path = g.dijkstra_shortest_path ("P12345", "Q98765")

# 导出所有节点对的最短路径，输出格式与算法一中格式一致
g.dijkstra_export_all_paths( "dijkstra_results.txt")


```

### 3.Bellman-Ford算法
**特点**：单源最短路径，支持负权重
**时间复杂度**：O(mn)
**使用实例**
```python
# 计算从指定起点到所有节点的最短路径
paths = g.bellmanford_shortest_paths( "P12345")

# 计算指定起点到终点的最短路径
path = g.bellmanford_shortest_path( "P12345", "Q98765")

# 导出结果到文件
g.bellmanford_export_all_paths(paths, "bellmanford_results.txt")

```


### 4.Johnson算法
**特点**：全源最短路径，支持负权重
**时间复杂度**：O(mn log n)
**使用实例**
```
python
# 计算从指定起点到所有节点的最短路径
paths = g.johnson_shortest_paths( "P12345")

# 计算指定起点到终点的最短路径
path = g.johnson_shortest_path("P12345", "Q98765")

# 导出所有节点对的最短路径
g.johnson_export_all_paths( "johnson_results.txt")
```

### 算法选择建议
| 场景                 | 推荐算法              |
|----------------------|-----------------------|
| 小型网络(<100节点)   | Floyd-Warshall       |
| 大型网络单源查询     | Dijkstra             |
| 大型网络全源查询     | Johnson              |
| 含负权重的网络       | Bellman-Ford或Johnson|


---

## 3.文件操作
### 1.操作实例
```python
#从文件生成图
graph = create_graph_from_file("protein_data.txt")
# 四种算法导出所有节点对的最短路径
graph.johnson_export_all_paths( "johnson_results.txt")
graph.bellmanford_export_all_paths(paths, "bellmanford_results.txt")
graph.dijkstra_export_all_paths( "dijkstra_results.txt")
graph.floyd_warshall_export( "floyd_results.txt")
```

### 2.输入文件格式
```
protein1 protein2 combined_score
P12345 Q98765 850
Q98765 R54321 920
...
```

### 3.输出文件格式
**所有算法导出的结果文件使用统一格式：**
```
node1 node2 total_weight path
P12345 Q98765 300 P12345->P54321->Q98765
...
```

### 4.文件转换关系示意图
```mermaid
graph LR
    A[原始数据文件] --> B[Graph对象]
    B --> C[算法处理]
    C --> D[结果文件]
```

---

## 4.图随机生成器
### 1.使用实例
```python
# 生成包含100个节点的稀疏图（边数约为最大可能边数的1%）
random_graph = create_undirected_connected_graph(num_nodes=100, sparse=0.01)

# 生成包含50个节点的较稠密图（边数约为最大可能边数的10%）
dense_graph = create_undirected_connected_graph(num_nodes=50, sparse=0.1)


```

### 2. 参数说明

| 参数         | 类型      | 默认值 | 说明                                                 |
|--------------|-----------|---------|------------------------------------------------------|
| **num_nodes** | int       | 100     | 节点的数量                                         |
| **sparse**    | float     | 0.01    | 图的稀疏度（值范围为0.0到1.0），表示边的稠密程度  |

### 3.节点命名规则

生成的节点名称遵循以下格式：
**0000.XXXXXXYYYY**

其中：
- `XXXXXX`：6位大写字母（A-Z）
- `YYYY`：4位数字（0000-9999）

示例节点名称：
- `0000.ABCDEF0123`
- `0000.XYZABC0456`





---
## 5. 综合示例
```python
from graph import Graph
from graph_generate import create_undirected_connected_graph

# 1. 使用随机网络生成器创建随机蛋白质网络
protein_net = create_undirected_connected_graph(num_nodes=50, sparse=0.05)

# 2. 导出原始网络,形成数据
protein_net.output_graph("random_network.txt")

# 3. 使用Floyd-Warshall计算全源最短路径
protein_net.floyd_warshall_export("floyd_paths.txt")

# 4. 使用Dijkstra计算特定蛋白质到所有其他蛋白质的路径
start_node = "0000.ABCDEF1234"  # 选择一个随机节点
dijkstra_paths = protein_net.dijkstra_shortest_paths(start_node)
with open("dijkstra_paths.txt", "w") as f:
    f.write("node1 node2 total_weight path\n")
    for path in dijkstra_paths:
        f.write(f"{path}\n")

# 5. 查找两个特定蛋白质间的最短路径
end_node = "0000.GHIJKL5678"  # 选择另一个随机节点
short_path = protein_net.dijkstra_shortest_path(start_node, end_node)
print(f"从 {start_node} 到 {end_node} 的最短路径: {short_path}")


```
---
## 6. 注意事项

### 权重转换
蛋白质间的"距离"由结合分数转换得到：  
`weight = 1000 - combined_score`

### 路径不存在
当路径不存在时，结果中权重显示为 **`"inf"`**

### 大型网络限制
节点数 > 1000 时：
- 避免使用 Floyd-Warshall 算法
- 优先使用 Dijkstra 或 Johnson 算法

### 负权重处理
当图中存在负权重时：
- 使用 Bellman-Ford 或 Johnson 算法
- Dijkstra 算法不适用

