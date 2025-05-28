# 四个算法
from graph import Graph
import numpy as np

#Floyd-Warshall
def floyd_warshall(graph:Graph):
    n = len(graph.adj)
    # 深拷贝原始图，避免修改输入数据,以及利用graph中储存的邻接表生成方便该算法编写的邻接矩阵
    path_matrix=[[None]*n for i in range(n)]#每个点对之间生成一个路径矩阵
    dist=np.zeros((n,n),dtype=float)#初始化距离矩阵，距离全部设置为零，并在下一步更新距离，

    #邻接矩阵的初始化
    node_to_id={}#初始化两个节点编号的字典，方便形成邻接矩阵
    id_to_node={}
    i=0
    for node in graph.adj.keys():
        node_to_id[node]=i
        id_to_node[i]=node
        i+=1
    for node in graph.adj.keys():
        its_index=node_to_id[node]
        for index in range(n):
            if index==its_index:
                pass
            else:
                next_node=id_to_node[index]
                if next_node in graph.adj[node].keys():
                    weight=graph.adj[node][next_node]
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





    



#Dijkstra


#Bellman-Ford


#Johnson
