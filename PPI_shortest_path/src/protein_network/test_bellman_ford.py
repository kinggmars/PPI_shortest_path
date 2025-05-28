from graph import Graph
from algorithms import bellman_ford, write_bf_results
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
        
        # 测试用例2: 带负权边但无负环的图
        self.graph2 = Graph()
        self.graph2.add_edge('A', 'B', 1)
        self.graph2.add_edge('B', 'C', -2)
        self.graph2.add_edge('C', 'D', -1)
        self.graph2.add_edge('D', 'A', 4)
        
        # 测试用例3: 存在负权环的图
        self.graph3 = Graph()
        self.graph3.add_edge('A', 'B', 1)
        self.graph3.add_edge('B', 'C', -2)
        self.graph3.add_edge('C', 'A', -1)
        
        # 测试用例4: 单个节点的图
        self.graph4 = Graph()
        self.graph4.add_node('A')
        
        # 测试用例5: 不连通图
        self.graph5 = Graph()
        self.graph5.add_edge('A', 'B', 1)
        self.graph5.add_edge('C', 'D', 2)
        
        # 测试用例6: 完全图
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
    
    def test_negative_weights_no_cycle(self):
        """测试带负权边但无负环的情况"""
        results = bellman_ford(self.graph2, 'A')
        expected = [
            'A A 0 A',
            'A B 1 A->B',
            'A C -1 A->B->C',
            'A D -2 A->B->C->D'
        ]
        self.assertCountEqual(results, expected)
    
    def test_negative_weight_cycle(self):
        """测试存在负权环的情况"""
        with self.assertRaises(ValueError):
            bellman_ford(self.graph3, 'A')
    
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
        
        # 验证文件内容
        with open("test_output.txt", 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
        self.assertEqual(lines, results)
        
        # 无效路径测试
        self.assertFalse(write_bf_results(results, "/invalid_path/output.txt"))

if __name__ == '__main__':
    unittest.main(verbosity=2)
