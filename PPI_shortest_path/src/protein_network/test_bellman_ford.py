from graph import Graph
from algorithms import bellman_ford, write_bf_results, get_shortest_path, get_result_by_end_node
import unittest

class TestBellmanFord(unittest.TestCase):
    def setUp(self):
        # 测试用例1: 简单图
        self.graph1 = Graph()
        self.graph1.add_edge('A', 'B', 1)
        self.graph1.add_edge('A', 'C', 4)
        self.graph1.add_edge('B', 'C', 2)
        self.graph1.add_edge('B', 'D', 5)
        self.graph1.add_edge('C', 'D', 1)
        
        # 测试用例2: 单个节点的图
        self.graph4 = Graph()
        self.graph4.add_node('A')
        
        # 测试用例3: 不连通图
        self.graph5 = Graph()
        self.graph5.add_edge('A', 'B', 1)
        self.graph5.add_edge('C', 'D', 2)
        
        # 测试用例4: 完全图
        self.graph6 = Graph()
        self.graph6.add_edge('A', 'B', 1)
        self.graph6.add_edge('A', 'C', 1)
        self.graph6.add_edge('A', 'D', 1)
        self.graph6.add_edge('B', 'A', 1)
        self.graph6.add_edge('B', 'C', 1)
        self.graph6.add_edge('B', 'D', 1)
        self.graph6.add_edge('C', 'A', 1)
        self.graph6.add_edge('C', 'B', 1)
        self.graph6.add_edge('C', 'D', 1)
        self.graph6.add_edge('D', 'A', 1)
        self.graph6.add_edge('D', 'B', 1)
        self.graph6.add_edge('D', 'C', 1)
    
    def test_normal_case(self):
        """测试正常情况下的最短路径计算"""
        results = bellman_ford(self.graph1, 'A')
        expected = [
            'A A 0 A',
            'A B 1 A->B',
            'A C 3 A->B->C',
            'A D 4 A->B->C->D'
        ]
        self.assertCountEqual(results, expected)
    
    def test_single_node(self):
        """测试只有一个节点的情况"""
        results = bellman_ford(self.graph4, 'A')
        self.assertEqual(results, ['A A 0 A'])
    
    def test_disconnected_graph(self):
        """测试不连通图的情况"""
        results = bellman_ford(self.graph5, 'A')
        expected = [
            'A A 0 A',
            'A B 1 A->B',
            'A C inf',
            'A D inf'
        ]
        self.assertCountEqual(results, expected)
    
    def test_complete_graph(self):
        """测试完全图的情况"""
        results = bellman_ford(self.graph6, 'A')
        expected = [
            'A A 0 A',
            'A B 1 A->B',
            'A C 1 A->C',
            'A D 1 A->D'
        ]
        self.assertCountEqual(results, expected)
    
    def test_invalid_start_node(self):
        """测试无效起始节点"""
        with self.assertRaises(KeyError):
            bellman_ford(self.graph1, 'Z')

    def test_file_output(self):
        """测试文件输出功能"""
        results = [
            'A B 3 A->C->B',
            'A C 2 A->C',
            'A D 5 A->C->D'
        ]
        
        # 正常写入测试
        self.assertTrue(write_bf_results(results, "test_output.txt"))
        
        # 验证文件内容（应包含标题行）
        with open("test_output.txt", 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
        expected = ["node1 node2 total_weight path"] + results  # 添加标题行
        self.assertEqual(lines, expected)  # 比较完整内容
        
        # 无效路径测试
        self.assertFalse(write_bf_results(results, "/invalid_path/output.txt"))

    def test_get_shortest_path(self):
        """测试从起始节点到目标节点的最短路径查询"""
        # 测试用例1: 正常可达路径
        result = get_shortest_path(self.graph1, 'A', 'D')
        self.assertEqual(result, ['A D 4 A->B->C->D'])
        
        # 测试用例2: 不可达路径
        result = get_shortest_path(self.graph5, 'A', 'C')
        self.assertEqual(result, ['A C inf'])
        
        # 测试用例3: 目标节点不存在
        result = get_shortest_path(self.graph1, 'A', 'Z')
        self.assertEqual(result, ['A Z inf'])

    def test_get_result_by_end_node(self):
        """测试从结果列表中提取特定目标节点的路径"""
        results = [
            'A A 0 A',
            'A B 1 A->B',
            'A C 3 A->B->C',
            'A D 4 A->B->C->D'
        ]
        
        # 测试用例1: 存在的目标节点
        self.assertEqual(get_result_by_end_node(results, 'C'), ['A C 3 A->B->C'])
        
        # 测试用例2: 不存在的目标节点
        self.assertEqual(get_result_by_end_node(results, 'Z'), ['A Z inf'])
        
        # 测试用例3: 不可达的节点
        results_disconnected = [
            'A A 0 A',
            'A B 1 A->B',
            'A C inf',
            'A D inf'
        ]
        self.assertEqual(get_result_by_end_node(results_disconnected, 'C'), ['A C inf'])

if __name__ == '__main__':
    unittest.main(verbosity=2)
