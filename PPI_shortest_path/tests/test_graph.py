# 测试图类能不能实现正常功能
import sys
import os

# 手动添加包路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..//src')))

# 现在可以正常导入
from protein_network.graph import Graph

if __name__ == '__main__':
    # 用原始数据初始化图
    initial_data = {
        '1': {'2': 2, '4': 1},
        '2': {'4': 3, '5': 11},
        '3': {'1': 4, '6': 5},
        '4': {'3': 2, '6': 8, '7': 4, '5': 2},
        '5': {'7': 6},
        '7': {'6': 1}
    }

    g = Graph(initial_data)

    # 打印图的邻接表
    print("--- 初始图结构 ---")
    print(g)

    # 操作示例
    print("\n--- 操作测试 ---")
    print("节点'1'的邻接节点:", list(g.get_neighbors('1')))  # [('2', 2), ('4', 1)]
    print("边'4'->'5'的权重:", g.get_edge_weight('4', '5'))  # 2
    print("是否存在边'7'->'6':", g.has_edge('7', '6'))      # True

    # 添加新边
    g.add_edge('6', '3', weight=7)
    print("\n添加边 6->3 后:")
    print(g.adj['6'])  # {'3': 7}

    # 删除节点'5'
    g.remove_node('5')
    print("\n删除节点'5'后:")
    print(g)

    # 将结果输出
    g.output_graph('test.txt')
