import os
import pickle
import platform
import sys

from pr2_new_2 import GetMultipleMetrixMethod_2

system = platform.system()
sys.path.append(r"D:\python\malwareTest\pr2_new_2")
if system == "Linux":
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr2_new_3")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr2_new_2")


def search_nodes_by_feature(graph, feature_name,feature_value):
    """在图中搜索具有指定特征的节点"""
    matching_nodes = []
    for node in graph.nodes():
        identity=''
        identity+=str(node.permission)
        identity+=str(node.sensitiveApi)
        identity+=str(node.suspiciousApi)
        identity+=str(node.component)
        identity+=str(node.url)

        if feature_name.lower() in identity.lower():
            print(identity.lower())
            matching_nodes.append({
                'node_name': node.name,
                'feature_value': feature_value,
                'node_object': node
            })
    return matching_nodes
graph_dir = '/home/changxiaosong/python/malwareTest/pr2' + os.sep + \
            'decompiled_java' + os.sep + 'graph_tmp' + os.sep + 'gra_old'
graph_file = r'C:\Users\chang\Desktop' + os.sep + str(9036) + '.pkl'
with open(graph_file, 'rb') as f:
    graph = pickle.load(f)
matching_nodes=search_nodes_by_feature(graph,'webkit',10)

print(len(matching_nodes))