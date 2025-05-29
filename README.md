# 蛋白质网络分析工具
- 本文主要介绍代码的具体实现逻辑
- 具体的函数调用方法详见用户手册
## 数据结构设计
### 邻接表的选择
本项目使用邻接表存储图结构，主要原因是：
- 空间效率高：蛋白质网络通常包含大量节点（蛋白质），但每个节点的连接相对稀疏。邻接矩阵的空间复杂度为O(n²)，而邻接表仅需O(n + e)，显著节省存储空间。
- 动态扩展性：方便动态添加/删除节点和边，适应蛋白质网络可能的变化。
### 字典嵌套字典的实现
图结构采用Python的defaultdict嵌套字典实现：
```python
adj = {
    'ProteinA': {'ProteinB': 0.8, 'ProteinC': 1.2},
    'ProteinB': {'ProteinD': 0.5},
    # ...
}
```
嵌套字典实现的优势：
- 快速邻接查询：O(1) 时间复杂度获取节点的所有邻接节点
- 权重即时访问：直接通过adj[u][v]获取边权重
- 动态维护方便：添加/删除边操作仅需字典操作

## 核心功能模块
### 1.图生成函数
```python
def create_undirected_connected_graph(nodes, density)
```
- 功能：生成随机无向连通图
- 参数：
nodes: 节点数量
density: 边密度（0.0-1.0）
### 2.文件I/O函数
```python
def create_graph_from_file(filename)  # 从文件构建图
def output_graph(filename)            # 导出图到文件
```
文件格式示例：
```
ProteinA ProteinB 0.8
ProteinA ProteinC 1.2
ProteinB ProteinD 0.5
```
### 3.最短路径算法
#### 1. Floyd-Warshall 算法
```python
def floyd_warshall():
    初始化距离矩阵dist和路径矩阵path
    for k in 所有中间节点:
        for i in 所有起点:
            for j in 所有终点:
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    更新dist[i][j]和路径
    return 距离矩阵和路径信息
```
#### 适用场景：
- 小规模网络（节点数 < 500）
- 需要所有节点对的最短路径
- 允许处理负权边（但不能有负权环）
#### 2. Dijkstra 算法
```python
def dijkstra(start):
    初始化优先队列和距离字典
    while 堆非空:
        u = 弹出当前最近节点
        for 每个邻接节点v:
            if 通过u到达v更短:
                更新距离并加入堆
    返回最短路径树
```
#### 适用场景：
- 单源最短路径问题
- 图中无负权边
- 中等规模网络（节点数 < 10^5）
- 使用优先队列优化效率（时间复杂度O(m + n log n)）
#### 3. Bellman-Ford 算法
```python
def bellman_ford(start):
    初始化距离数组
    for i in 1 to n-1:
        for 每条边(u, v):
            if 松弛操作成功:
                更新距离
    检测负权环
    返回最短路径
```
#### 适用场景：
- 存在负权边的情况
- 需要检测负权环
- 单源最短路径问题
#### 4. Johnson 算法
```python
def johnson():
    添加虚拟节点并运行Bellman-Ford
    重新赋权所有边
    for 每个节点u:
        运行Dijkstra算法
    还原原始权重
    返回所有节点对最短路径
```
#### 适用场景：
- 大规模稀疏网络
- 需要所有节点对的最短路径
- 包含负权边但不含负权环
- 比Floyd-Warshall更适合大规模稀疏图
## 性能比较
通过实验分析不同算法在不同图结构下的表现：
```
![time_performance.png]
```
- 稠密图：Floyd-Warshall 表现稳定
- 稀疏图：Johnson 算法效率优势明显
- 单源查询：Dijkstra 时间复杂度最优
- 负权边：Bellman-Ford 是唯一选择
### 测试用例
测试所用的数据下载自string数据库
```
4909.protein.links.v12.0.txt
```
### 项目包含完整的测试套件：
- test_generate_read.py：验证图生成和文件I/O功能
- test_graph.py：测试图操作的核心逻辑
- test_performance.py：算法性能对比实验
- test_real_data: 用真实世界数据测试，输出两个点之间的最短路径


## 项目背景
本工具是《生物编程语言》课程的核心实践项目，旨在通过实际生物信息学场景（蛋白质相互作用网络）深入理解图论算法的工程实现。

## 参考文献
- 陈益富, 卢潇, 丁豪杰. 对Dijkstra算法的优化策略研究[J]. 计算机技术与发展, 2006(09).
- 韩伟一. 经典Bellman-Ford算法的改进及其实验评估[J]. 哈尔滨工业大学学报, 2012, 44(07).
- Stark C, Breitkreutz B-J, Reguly T, Boucher L, Breitkreutz A, Tyers M. BioGRID: a general repository for interaction datasets. *Nucleic acids research.* 2006;34(suppl 1):D535–D539.
- Wilson N. Human Protein Reference Database. *Nature Reviews Molecular Cell Biology.* 2004;5(1):4–4.
