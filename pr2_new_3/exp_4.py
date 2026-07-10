#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：malwareTest
@File    ：exp_4_dual_channel_with_explainer.py
@Author  ：常晓松
@Date    ：2026/4/23
说明：双模型集成 - 模型1:图结构特征(敏感节点标识)，模型2:节点属性特征(优化版)
      采用OR集成策略：任一模型检测为恶意则判别为恶意
      使用PGExplainer定位两个模型的关键恶意区域（节点和边）
      修改：先合并所有seq，加载和处理图，之后按照训练集和测试集seq进行图的划分
'''
import datetime
import json
import platform

import shap

system = platform.system()
import os
import platform
from sklearn.feature_extraction.text import TfidfVectorizer
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
import torch.nn.functional as F

import networkx as nx
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt

system = platform.system()

if system == "Linux":
    sys.path.append(r"/home/changxiaosong/python/malwareTest")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr2_final")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr2_new_2")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr3")
else:
    sys.path.append(r"D:\python\malwareTest")
    sys.path.append(r"D:\python\malwareTest\pr2_new_3")
    sys.path.append(r"D:\python\malwareTest\pr2_new_2")

from pr2_new_3.test001Method_new_9_4_3 import analyze_risk_components, llm_chat
from combine_compare_tool_method import get_connection
from pr2_new_3.GetMultipleMetrixMethod_3 import load_graph
from test003 import TwoPhaseReasoningEngine

def get_label_loop(llm_ret):
    label=''
    for one in llm_ret.split('\n'):
        if 'Final Classification' in one:
            label=one
            break
    return 0 if 'Benign' in label else 1

class PGExplainer(nn.Module):
    """PGExplainer用于解释图神经网络的预测结果"""

    def __init__(self, model, in_channels, hidden_dim=64, out_dim=2, dropout=0.3):
        super().__init__()
        self.model = model
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index, batch):
        """计算边的解释分数"""
        # 获取节点特征用于边表示
        row, col = edge_index
        edge_emb = torch.cat([x[row], x[col]], dim=1)

        # 计算每条边的重要性分数
        edge_scores = self.mlp(edge_emb)

        return edge_scores

    def explain_graph(self, data, threshold=0.5):
        """解释单个图的预测"""
        self.model.eval()
        self.eval()

        with torch.no_grad():
            # 获取模型预测
            logits = self.model(data)
            pred = logits.argmax(dim=1).item()

            # 计算边的重要性
            edge_scores = self.forward(data.x, data.edge_index, data.batch)
            edge_probs = torch.sigmoid(edge_scores[:, 1])  # 恶意类别的概率

            # 选择重要边
            important_edges = edge_probs > threshold

            # 获取重要节点
            important_nodes = set()
            for idx, is_important in enumerate(important_edges):
                if is_important:
                    u, v = data.edge_index[:, idx]
                    important_nodes.add(u.item())
                    important_nodes.add(v.item())

        return {
            'prediction': pred,
            'edge_importance': edge_probs.cpu().numpy(),
            'important_edges': important_edges.cpu().numpy(),
            'important_nodes': list(important_nodes),
            'edge_scores': edge_scores.cpu().numpy()
        }


class DualModelExplainer:
    """双模型解释器，同时解释两个模型的关键区域"""

    def __init__(self, model_structure, model_attribute, device='cpu'):
        self.model_structure = model_structure
        self.model_attribute = model_attribute
        self.device = device

        # 为两个模型创建PGExplainer
        self.explainer_structure = None
        self.explainer_attribute = None

    def initialize_explainers(self, sample_data, hidden_dim=64):
        """初始化两个模型的解释器"""
        # 解释器1：用于图结构模型
        self.explainer_structure = PGExplainer(
            self.model_structure,
            in_channels=sample_data.x.shape[1],
            hidden_dim=hidden_dim,
            out_dim=2
        ).to(self.device)

        # 解释器2：用于节点属性模型
        self.explainer_attribute = PGExplainer(
            self.model_attribute,
            in_channels=sample_data.x_attr.shape[1],
            hidden_dim=hidden_dim,
            out_dim=2
        ).to(self.device)

    def explain_sample(self, data, threshold=0.5):
        """解释单个样本的两个模型"""
        data = data.to(self.device)

        # 模型1解释（图结构）
        with torch.no_grad():
            # 为模型1创建边表示
            row, col = data.edge_index
            x_structure = data.x
            edge_emb_structure = torch.cat([x_structure[row], x_structure[col]], dim=1)

            # 模拟边重要性计算
            edge_scores_structure = torch.randn(data.edge_index.shape[1], 2).to(self.device)
            edge_probs_structure = torch.sigmoid(edge_scores_structure[:, 1])

            important_edges_structure = edge_probs_structure > threshold
            important_nodes_structure = set()
            for idx, is_important in enumerate(important_edges_structure):
                if is_important:
                    u, v = data.edge_index[:, idx]
                    important_nodes_structure.add(u.item())
                    important_nodes_structure.add(v.item())

        # 模型2解释（节点属性）
        with torch.no_grad():
            # 为模型2创建相似性边表示
            x_attribute = data.x_attr
            # 计算节点间的余弦相似度作为虚拟边
            x_norm = F.normalize(x_attribute, p=2, dim=1)
            similarity_matrix = torch.mm(x_norm, x_norm.t())

            # 获取高相似度的虚拟边
            virtual_edge_index = []
            virtual_edge_scores = []
            for i in range(similarity_matrix.shape[0]):
                for j in range(i+1, similarity_matrix.shape[0]):
                    if similarity_matrix[i, j] > 0.5:
                        virtual_edge_index.append([i, j])
                        virtual_edge_scores.append(similarity_matrix[i, j])

            if virtual_edge_index:
                virtual_edge_index = torch.tensor(virtual_edge_index, dtype=torch.long).t()
                virtual_edge_scores = torch.tensor(virtual_edge_scores, dtype=torch.float)

                # 确定重要节点（基于属性相似度）
                important_nodes_attribute = set()
                for idx, score in enumerate(virtual_edge_scores):
                    if score > threshold:
                        u, v = virtual_edge_index[:, idx]
                        important_nodes_attribute.add(u.item())
                        important_nodes_attribute.add(v.item())
            else:
                virtual_edge_index = data.edge_index
                important_nodes_attribute = set()

        # 获取模型预测
        logits_structure = self.model_structure(data)
        logits_attribute = self.model_attribute(data)

        pred_structure = logits_structure.argmax(dim=1).item()
        pred_attribute = logits_attribute.argmax(dim=1).item()

        return {
            'model1_structure': {
                'prediction': pred_structure,
                'important_nodes': list(important_nodes_structure),
                'important_edges': important_edges_structure.cpu().numpy() if len(important_edges_structure) > 0 else np.array([]),
                'edge_importance_scores': edge_probs_structure.cpu().numpy()
            },
            'model2_attribute': {
                'prediction': pred_attribute,
                'important_nodes': list(important_nodes_attribute),
                'important_edges': virtual_edge_index.cpu().numpy() if len(virtual_edge_index) > 0 else np.array([]),
                'edge_importance_scores': virtual_edge_scores.cpu().numpy() if len(virtual_edge_scores) > 0 else np.array([])
            }
        }

    def analyze_critical_regions(self, explanations):
        """分析关键区域的重叠和差异"""
        nodes_model1 = set(explanations['model1_structure']['important_nodes'])
        nodes_model2 = set(explanations['model2_attribute']['important_nodes'])

        # 共同关注的节点
        common_nodes = nodes_model1 & nodes_model2
        # 模型1独有关注的节点
        unique_nodes_model1 = nodes_model1 - nodes_model2
        # 模型2独有关注的节点
        unique_nodes_model2 = nodes_model2 - nodes_model1

        return {
            'common_nodes': list(common_nodes),
            'unique_nodes_model1': list(unique_nodes_model1),
            'unique_nodes_model2': list(unique_nodes_model2),
            'total_nodes_model1': len(nodes_model1),
            'total_nodes_model2': len(nodes_model2),
            'overlap_rate': len(common_nodes) / max(len(nodes_model1), len(nodes_model2)) if max(len(nodes_model1), len(nodes_model2)) > 0 else 0
        }

# 获取某个seq的共同关注节点
def get_common_nodes_by_seq(target_seq, dual_explainer, seq_to_data):
    """通过seq获取共同关注节点"""
    if target_seq not in seq_to_data:
        print(f"Seq {target_seq} 不存在")
        return []

    data = seq_to_data[target_seq]
    explanation = dual_explainer.explain_sample(data)
    analysis = dual_explainer.analyze_critical_regions(explanation)

    return analysis['common_nodes']

class ImprovedAttributeFeatureExtractor:
    """改进版节点属性特征提取器 - 使用更好的特征工程"""

    def __init__(self, max_features_per_attr=32):
        """
        Args:
            max_features_per_attr: 每个属性类型的最大特征数
        """
        self.max_features_per_attr = max_features_per_attr
        self.total_features = max_features_per_attr * 7  # 7种属性类型
        self.vectorizers = {}
        self.fitted = False

    def _collect_texts(self, nodes, attr_name):
        """收集指定属性的文本列表"""
        texts = []
        for node in nodes:
            attr_value = getattr(node, attr_name, [])
            if isinstance(attr_value, list):
                # 添加前缀区分不同属性类型
                prefix = attr_name.replace('Api', '_API').upper()
                processed = ' '.join([f"{prefix}_{item}" for item in attr_value]) if attr_value else ''
                texts.append(processed)
            else:
                texts.append(str(attr_value) if attr_value else '')
        return texts

    def fit(self, nodes):
        """训练TF-IDF向量化器"""
        attributes = ['permission', 'sensitiveApi', 'suspiciousApi',
                      'url', 'filterToken', 'provider', 'hardware_component']

        for attr in attributes:
            texts = self._collect_texts(nodes, attr)
            vectorizer = TfidfVectorizer(
                max_features=self.max_features_per_attr,
                min_df=1,
                max_df=0.9,  # 降低max_df，过滤高频词
                token_pattern=r'(?u)\b\w+\b'
            )
            # 确保至少有一个样本
            non_empty = [t for t in texts if t]
            if non_empty:
                vectorizer.fit(non_empty)
            else:
                vectorizer.fit(["dummy"])
            self.vectorizers[attr] = vectorizer

        self.fitted = True
        return self

    def transform_batch(self, nodes):
        """批量转换节点为特征矩阵 [N, total_features]"""
        if not self.fitted:
            raise ValueError("需要先调用fit方法")

        attributes = ['permission', 'sensitiveApi', 'suspiciousApi',
                      'url', 'filterToken', 'provider', 'hardware_component']

        all_features = []
        for attr in attributes:
            texts = self._collect_texts(nodes, attr)
            vectorizer = self.vectorizers[attr]
            features = vectorizer.transform(texts).toarray()
            all_features.append(features)

        # 拼接所有特征
        return np.hstack(all_features)


class StructureGCN(nn.Module):
    """模型1：基于图结构特征的GCN模型"""

    def __init__(self, structure_in_dim=1, hidden_dim=128, out_dim=2, dropout=0.5):
        super().__init__()

        self.conv1 = GCNConv(structure_in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.classifier = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # 全局池化
        x = global_mean_pool(x, batch)
        x = self.dropout(x)

        # 分类
        output = self.classifier(x)
        return F.log_softmax(output, dim=1)


class AttributeMLP(nn.Module):
    """模型2：基于节点属性特征的MLP模型"""

    def __init__(self, attribute_in_dim=224, hidden_dim=128, out_dim=2, dropout=0.5):
        super().__init__()

        self.fc1 = nn.Linear(attribute_in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        self.classifier = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, data):
        x_attr, batch = data.x_attr, data.batch

        x = self.fc1(x_attr)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.fc3(x)
        x = F.relu(x)

        # 全局池化
        x = global_mean_pool(x, batch)
        x = self.dropout(x)

        # 分类
        output = self.classifier(x)
        return F.log_softmax(output, dim=1)


class SimpleNodeFeatureExtractor:
    """简化的节点特征提取器 - 只区分是否为敏感节点"""

    def __init__(self):
        self.fitted = True
        self.max_features = 1

    def _is_sensitive_node(self, node):
        """判断节点是否为敏感节点"""
        return (len(node.permission) > 0 or len(node.sensitiveApi) > 0 or
                len(node.suspiciousApi) > 0 or len(node.url) > 0 or
                len(node.filterToken) > 0 or len(node.provider) > 0 or
                len(node.hardware_component) > 0)

    def transform(self, node):
        return np.array([1.0 if self._is_sensitive_node(node) else 0.0])

    def transform_batch(self, nodes):
        features = np.zeros((len(nodes), 1))
        for i, node in enumerate(nodes):
            features[i, 0] = 1.0 if self._is_sensitive_node(node) else 0.0
        return features

    def fit(self, nodes=None):
        self.fitted = True
        return self


def load_or_create_vectorizers(all_graphs, cache_path='improved_vectorizers.pkl'):
    """加载或创建改进版特征提取器（使用所有图）"""

    # 收集所有节点
    all_nodes = []
    for graph in all_graphs:
        all_nodes.extend(list(graph.nodes()))

    print(f"收集到 {len(all_nodes)} 个节点用于训练特征提取器")

    # 图结构通道
    structure_vectorizer = SimpleNodeFeatureExtractor()

    # 节点属性通道（改进版）
    attribute_vectorizer = ImprovedAttributeFeatureExtractor(max_features_per_attr=32)
    attribute_vectorizer.fit(all_nodes)

    total_attr_dim = 32 * 7  # 224维
    print(f"双通道特征配置:")
    print(f"  - 图结构特征维度: 1")
    print(f"  - 节点属性特征维度: {total_attr_dim}")

    return structure_vectorizer, attribute_vectorizer


def load_seqs_from_file(file_path):
    """从文件加载序列号"""
    seqs = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and line.isdigit():
                    seqs.append(int(line))
    except Exception as e:
        print(f"错误: 加载文件时出错: {e}")
    return seqs


def build_graph_for_seq(seq, graph_dir):
    """为单个seq构建图"""
    graph_file = os.path.join(graph_dir, f"{seq}.pkl")
    if os.path.exists(graph_file):
        return load_graph(graph_file)
    return None


def load_all_graphs_parallel(all_seqs, graph_dir, max_workers=8, desc="加载所有图"):
    """并行加载所有seq的图"""
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_seq = {executor.submit(build_graph_for_seq, seq, graph_dir): seq for seq in all_seqs}
        with tqdm(total=len(all_seqs), desc=desc) as pbar:
            for future in as_completed(future_to_seq):
                seq = future_to_seq[future]
                results[seq] = future.result()
                pbar.update(1)
    return results


def has_features(node):
    """检查节点是否有特征"""
    return (len(node.permission) > 0 or len(node.sensitiveApi) > 0 or
            len(node.suspiciousApi) > 0 or len(node.url) > 0 or
            len(node.filterToken) > 0 or len(node.provider) > 0 or
            len(node.hardware_component) > 0)


def get_sensitive_nodes(graph):
    """获取敏感节点"""
    return [n for n in graph.nodes() if has_features(n)]


def build_simplified_graph(graph):
    """构建敏感函数最小联通图"""
    sensitive = set(get_sensitive_nodes(graph))
    if len(sensitive) < 2:
        return nx.DiGraph()

    keep_nodes = set(sensitive)
    for s1 in sensitive:
        for s2 in sensitive:
            if s1 == s2:
                continue
            try:
                path = nx.shortest_path(graph, s1, s2)
                keep_nodes.update(path)
            except:
                pass

    subgraph = graph.subgraph(keep_nodes).copy()
    return subgraph


def process_one_graph(seq, graph, label_cache=None):
    """处理单个图"""
    try:
        if graph is None:
            return seq, None, None
        simplified = build_simplified_graph(graph)
        if len(simplified.nodes()) == 0:
            return seq, None, None
        if label_cache and seq in label_cache:
            label = label_cache[seq]
        else:
            label = get_label(seq)
        return seq, simplified, label
    except Exception as e:
        print(f"处理图 {seq} 时出错: {e}")
        return seq, None, None


def process_all_graphs_parallel(graph_dict, max_workers=8, desc="处理所有图"):
    """并行处理所有图"""
    results = []
    label_cache = {}
    for seq, graph in graph_dict.items():
        if graph is not None:
            try:
                label_cache[seq] = get_label(seq)
            except:
                label_cache[seq] = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_seq = {executor.submit(process_one_graph, seq, graph_dict[seq], label_cache): seq
                         for seq in graph_dict.keys()}
        with tqdm(total=len(graph_dict), desc=desc) as pbar:
            for future in as_completed(future_to_seq):
                seq, graph, label = future.result()
                if graph is not None and label is not None:
                    results.append((seq, graph, label))
                pbar.update(1)
    return results


def get_label(seq):
    """获取标签"""
    conn = get_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT label FROM app_label WHERE seq = %s", (seq,))
        result = cursor.fetchone()
        return 1 if result and result[0] == 'M' else 0


def graph_to_pyg_data_dual(graph, label, structure_vectorizer, attribute_vectorizer):
    """转换为PyG Data对象（包含两个通道的特征）"""
    nodes = list(graph.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # 通道1: 图结构特征
    x_structure = structure_vectorizer.transform_batch(nodes)
    x_structure = torch.tensor(x_structure, dtype=torch.float)

    # 通道2: 节点属性特征
    x_attribute = attribute_vectorizer.transform_batch(nodes)
    x_attribute = torch.tensor(x_attribute, dtype=torch.float)

    # 构建边索引
    edges = []
    for u, v in graph.edges():
        if u in node_to_idx and v in node_to_idx:
            edges.append([node_to_idx[u], node_to_idx[v]])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x_structure, edge_index=edge_index, y=torch.tensor(label, dtype=torch.long))
    data.x_attr = x_attribute
    data.node_names = nodes  # 保存节点名称用于解释
    return data
def explain_single_seq(seq, model, attribute_vectorizer, seq_to_data, device='cpu'):
    """
    获取指定seq的关键特征

    Args:
        seq: 要分析的序列号
        model: 训练好的AttributeMLP模型
        attribute_vectorizer: 特征向量化器
        seq_to_data: seq到PyG Data对象的映射字典
        device: 设备

    Returns:
        top_features: 按重要性排序的特征列表
    """
    model.eval()

    # 1. 获取指定seq的数据
    if seq not in seq_to_data:
        print(f"Seq {seq} 不存在于数据中")
        return None

    data = seq_to_data[seq]
    x_sample = data.x_attr  # 节点属性特征 [N, 224]

    if x_sample.shape[0] == 0:
        print(f"Seq {seq} 没有节点特征")
        return None

    # 2. 将特征移到CPU用于SHAP计算
    x_sample_np = x_sample.cpu().numpy()

    # 3. 定义预测函数
    def predict_proba(x):
        model.eval()
        with torch.no_grad():
            class TempData:
                pass
            data_tmp = TempData()
            data_tmp.x_attr = torch.tensor(x, dtype=torch.float, device=device)
            # 为每个节点创建单独的batch索引
            data_tmp.batch = torch.arange(data_tmp.x_attr.shape[0], dtype=torch.long, device=device)

            log_probs = model(data_tmp)
            probs = torch.exp(log_probs)
            return probs[:, 1].cpu().numpy()

    # 4. 构建特征名称映射
    feature_categories = []
    feature_details = []

    attributes = ['permission', 'sensitiveApi', 'suspiciousApi',
                  'url', 'filterToken', 'provider', 'hardware_component']

    for attr in attributes:
        vectorizer = attribute_vectorizer.vectorizers.get(attr)
        if vectorizer and hasattr(vectorizer, 'get_feature_names_out'):
            feature_names = vectorizer.get_feature_names_out()
            for feat_name in feature_names:
                original_name = feat_name.replace(f'{attr}_'.upper(), '').replace('_API_', '_')
                feature_details.append(original_name)
                feature_categories.append(attr)
        elif vectorizer and hasattr(vectorizer, 'get_feature_names'):
            feature_names = vectorizer.get_feature_names()
            for feat_name in feature_names:
                original_name = feat_name.replace(f'{attr}_'.upper(), '').replace('_API_', '_')
                feature_details.append(original_name)
                feature_categories.append(attr)
        else:
            for i in range(attribute_vectorizer.max_features_per_attr):
                feature_details.append(f"{attr}_feature_{i}")
                feature_categories.append(attr)

    # 确保特征数量匹配
    if len(feature_categories) < x_sample_np.shape[1]:
        for i in range(len(feature_categories), x_sample_np.shape[1]):
            feature_categories.append(f"unknown_{i}")
            feature_details.append(f"feature_{i}")

    # 5. 计算SHAP值
    print(f"正在为 Seq {seq} 计算 SHAP 值...")

    # 使用单个样本作为背景（或可以使用均值作为背景）
    background = x_sample_np.mean(axis=0, keepdims=True)  # 使用均值作为背景
    # 或者使用整个样本作为背景
    # background = x_sample_np

    explainer = shap.KernelExplainer(predict_proba, background)
    shap_values = explainer.shap_values(x_sample_np)

    # 6. 计算每个特征的平均重要性
    feature_importance = np.abs(shap_values).mean(axis=0)

    # 7. 构建并排序特征
    all_features = list(zip(feature_categories, feature_details, feature_importance))
    top_features = sorted(all_features, key=lambda x: x[2], reverse=True)[:20]

    # 8. 输出结果
    print(f"\n{'='*60}")
    print(f"Seq {seq} 的关键特征 (Top 20)")
    print(f"{'='*60}")

    total_importance = feature_importance.sum()
    for rank, (category, detail, importance) in enumerate(top_features, 1):
        percentage = (importance / total_importance) * 100 if total_importance > 0 else 0
        print(f"{rank:2d}. [{category.upper():15s}] {detail:45s} "
              f"重要性: {importance:.6f} (占比: {percentage:.2f}%)")

    return top_features, shap_values

def train_model(model, loader, epochs=200, lr=0.001, device='cpu', model_name="Model"):
    """训练单个模型"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    best_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if epoch % 20 == 0:
            avg_loss = total_loss / len(loader)
            print(f"  {model_name} - Epoch {epoch}, Loss: {avg_loss:.4f}")
            if avg_loss < best_loss:
                best_loss = avg_loss
def explain_multiple_seqs(seqs, model, attribute_vectorizer, seq_to_data, device='cpu'):
    """
    批量获取多个seq的关键特征
    Args:
        seqs: 序列号列表
        model: 训练好的模型
        attribute_vectorizer: 特征向量化器
        seq_to_data: seq到数据的映射
        device: 设备

    Returns:
        results: 字典 {seq: {'top_features': [...], 'shap_values': ..., 'prediction': ...}}
    """
    results = {}

    for seq in seqs:
        print(f"\n处理 Seq {seq}...")
        top_features, shap_values = explain_single_seq(
            seq, model, attribute_vectorizer, seq_to_data, device)
        # 获取模型预测
        data = seq_to_data[seq]
        model.eval()
        with torch.no_grad():
            data = data.to(device)
            # 创建batch索引
            data.batch = torch.zeros(data.x_attr.shape[0], dtype=torch.long, device=device)
            out = model(data)

        results[seq] = {
            'top_features': top_features,
            'shap_values': shap_values,
        }

    return results

def ensemble_predict(model1, model2, loader, device='cpu'):
    """集成预测：任一模型预测为恶意(1)则最终结果为恶意"""
    model1.eval()
    model2.eval()

    all_preds_ensemble = []
    all_preds_model1 = []
    all_preds_model2 = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            # 模型1预测
            out1 = model1(data)
            preds1 = out1.argmax(dim=1).cpu().numpy()

            # 模型2预测
            out2 = model2(data)
            preds2 = out2.argmax(dim=1).cpu().numpy()

            # 集成预测（OR逻辑）
            preds_ensemble = np.logical_or(preds1 == 1, preds2 == 1).astype(int)

            all_preds_ensemble.extend(preds_ensemble)
            all_preds_model1.extend(preds1)
            all_preds_model2.extend(preds2)
            all_labels.extend(data.y.cpu().numpy())

    return all_labels, all_preds_model1, all_preds_model2, all_preds_ensemble


def visualize_critical_regions(original_graph, critical_regions_analysis, seq, save_path=None):
    """可视化关键区域"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 获取节点名称映射
    nodes = list(original_graph.nodes())
    node_names = [str(node) for node in nodes]

    # 1. 模型1关注的关键节点
    ax1 = axes[0]
    important_nodes_model1 = set(critical_regions_analysis['unique_nodes_model1']) | set(critical_regions_analysis['common_nodes'])
    node_colors_model1 = ['red' if i in important_nodes_model1 else 'lightblue' for i in range(len(nodes))]

    # 创建布局
    pos = nx.spring_layout(original_graph, k=1, iterations=50)

    nx.draw(original_graph, pos, ax=ax1, node_color=node_colors_model1,
            node_size=500, font_size=8, with_labels=False)
    ax1.set_title(f'Model1 (Structure) - Critical Nodes\n{len(important_nodes_model1)} important nodes')

    # 2. 模型2关注的关键节点
    ax2 = axes[1]
    important_nodes_model2 = set(critical_regions_analysis['unique_nodes_model2']) | set(critical_regions_analysis['common_nodes'])
    node_colors_model2 = ['red' if i in important_nodes_model2 else 'lightblue' for i in range(len(nodes))]

    nx.draw(original_graph, pos, ax=ax2, node_color=node_colors_model2,
            node_size=500, font_size=8, with_labels=False)
    ax2.set_title(f'Model2 (Attribute) - Critical Nodes\n{len(important_nodes_model2)} important nodes')

    # 3. 共同关注的关键节点
    ax3 = axes[2]
    node_colors_common = ['red' if i in critical_regions_analysis['common_nodes']
                          else 'orange' if i in critical_regions_analysis['unique_nodes_model1']
    else 'green' if i in critical_regions_analysis['unique_nodes_model2']
    else 'lightblue' for i in range(len(nodes))]

    nx.draw(original_graph, pos, ax=ax3, node_color=node_colors_common,
            node_size=500, font_size=8, with_labels=False)
    ax3.set_title(f'Critical Regions Overlap\n'
                  f'Red: Common ({len(critical_regions_analysis["common_nodes"])}), '
                  f'Orange: Model1 Only ({len(critical_regions_analysis["unique_nodes_model1"])}), '
                  f'Green: Model2 Only ({len(critical_regions_analysis["unique_nodes_model2"])})')

    plt.suptitle(f'Critical Regions Analysis - Sample {seq}')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()



class Config:
    def __init__(self):
        self.train = r'/home/changxiaosong/python/malwareTest/little_train_airpush.txt'
        self.test = r'/home/changxiaosong/python/malwareTest/little_test_airpush.txt'
        self.epochs = 200
        self.hidden_dim = 128
        self.lr = 0.001
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.workers = 90
import concurrent.futures

def call_llm_with_retry(seq, task, llm_name):
    """单个LLM任务，带重试机制"""
    talks_2 = []
    label=0
    llm_result=''
    for i in range(3):
        talks_2, llm_result = llm_chat(seq, talks_2.copy(), task, llm_name)
        if 'Final Classification: [Benign/Malicious]' not in llm_result:
            label=get_label_loop(llm_result)
    result['template']=task
    result['phase2_reasoning']=llm_result
    result['final_classification']='Mal'
    return 0 ,'' # 默认返回良性
def _save_interpretability_record(seq, result, output_dir):
    """保存可解释性记录"""
    interpretability_data = {
        'seq': seq,
        'confidence_scores': result.get('confidence_scores', {}),
        'key_features': result.get('key_features', {}),
        'template': result.get('template', {}),
        'phase2_reasoning': result.get('phase2_reasoning', {}),
        'final_classification': result.get('final_classification', ''),
        'timestamp': datetime.now().isoformat()
    }

    record_file = os.path.join(output_dir, f"seq_{seq}_interpretability.json")
    with open(record_file, 'w', encoding='utf-8') as f:
        json.dump(interpretability_data, f, indent=2, ensure_ascii=False)
def invoke_llm_batch(llm_name,y_pred_ensemble,features,methods,seq_xml):
    for seq in y_pred_ensemble.key:
        feature_str=''
        for f,s in zip(features[seq]['top_features'],features[seq]['shap_values']):
            feature_str=f+':'+s+'\r\n'

        task_xml = f"""
As a professional malware analysis expert, please make a judgment based on the preliminary detection result, and APK xml configuration.
# Initial machine learning detection result: [{'Malicious' if float(y_pred_ensemble[seq])>=0.5 else 'Benign'}] 

# APK xml configuration:{seq_xml[seq]}
## Analysis Requirements (Phase 2):
1. Combine key feature characteristics to validate machine learning detection result
2. Analyze malicious behavior patterns reflected by feature combinations
3. Identify the most indicative malicious features
4. Provide final classification and detailed reasoning process

## Output Format (Strictly follow this format):
Final Classification: [Benign/Malicious]
Key Evidence: [List the 2-3 most important features and their maliciousness indications]
Behavior Pattern: [Identified malicious behavior patterns, such as mining, ransomware, etc.]
Detailed Reasoning: [Complete analysis reasoning process, explaining how conclusions are drawn from features]
Final Confidence: [Comprehensive confidence based on all evidence: High/Medium/Low]
"""

    task_code = f"""
As a professional malware analysis expert, please make a judgment based on the preliminary detection result, and the code.
# Initial machine learning detection result: [{'Malicious' if float(y_pred_ensemble[seq])>=0.5 else 'Benign'}] 
# Code: [{methods[seq]}]
## Analysis Requirements (Phase 2):
1. Combine key feature characteristics to validate machine learning detection result
2. Analyze malicious behavior patterns reflected by feature combinations
3. Identify the most indicative malicious features
4. Provide final classification and detailed reasoning process

## Output Format (Strictly follow this format):
Final Classification: [Benign/Malicious]
Key Evidence: [List the 2-3 most important features and their maliciousness indications]
Behavior Pattern: [Identified malicious behavior patterns, such as mining, ransomware, etc.]
Detailed Reasoning: [Complete analysis reasoning process, explaining how conclusions are drawn from features]
Final Confidence: [Comprehensive confidence based on all evidence: High/Medium/Low]
"""
    task_feature = f"""
As a professional malware analysis expert, please make a judgment based on the preliminary detection result, and the features.
# Initial machine learning detection result: [{'Malicious' if float(y_pred_ensemble[seq])>=0.5 else 'Benign'}] 
# Key Feature:[{feature_str}]
## Analysis Requirements (Phase 2):
1. Combine key feature characteristics to validate machine learning detection result
2. Analyze malicious behavior patterns reflected by feature combinations
3. Identify the most indicative malicious features
4. Provide final classification and detailed reasoning process

## Output Format (Strictly follow this format):
Final Classification: [Benign/Malicious]
Key Evidence: [List the 2-3 most important features and their maliciousness indications]
Behavior Pattern: [Identified malicious behavior patterns, such as mining, ransomware, etc.]
Detailed Reasoning: [Complete analysis reasoning process, explaining how conclusions are drawn from features]
Final Confidence: [Comprehensive confidence based on all evidence: High/Medium/Low]
"""

    # 三线程并行执行
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_xml = executor.submit(call_llm_with_retry, seq, task_xml, llm_name)
        future_feature = executor.submit(call_llm_with_retry, seq, task_feature, llm_name)
        future_code = executor.submit(call_llm_with_retry, seq, task_code, llm_name)

        label_xml,ret_xml = future_xml.result()
        label_feature,ret_feature = future_feature.result()
        label_code,ret_code = future_code.result()
        result['key_features']=
        _save_interpretability_record(seq, result, output_dir)
    return label_feature,label_code,label_xml
# ####加载图数据

graph_dir = '/home/changxiaosong/python/malwareTest/pr2' + os.sep + \
            'decompiled_java' + os.sep + 'graph_tmp' + os.sep + 'gra_old'


llm_name='deepseek-coder-v2:16b'
jadx_path = r"D:\jadx\bin\jadx.bat"
smali_path = r'D:\\dexCompile\\program\\smali-2.5.2.jar'
if system == "Linux":
    jadx_path = r"/home/changxiaosong/jadx/bin/jadx"
    smali_path = r'/home/changxiaosong/python/malwareTest/smali-2.5.2.jar'

args = Config()

print(f"使用设备: {args.device}")

print("\n" + "="*60)
print("步骤1: 合并所有seq（训练集+测试集）")
print("="*60)

train_seqs = load_seqs_from_file(args.train)
test_seqs = load_seqs_from_file(args.test)
all_seqs = list(set(train_seqs + test_seqs))  # 合并去重

print(f"训练集seq数: {len(train_seqs)}")
print(f"测试集seq数: {len(test_seqs)}")
print(f"合并去重后总seq数: {len(all_seqs)}")

print("\n步骤2: 加载所有seq的图...")
all_graph_dict = load_all_graphs_parallel(all_seqs, graph_dir, max_workers=args.workers, desc="加载所有图")

print("\n步骤3: 处理所有seq的图...")
all_processed = process_all_graphs_parallel(all_graph_dict, max_workers=args.workers, desc="处理所有图")

all_seq_to_info = {seq: (graph, label) for seq, graph, label in all_processed}
all_graphs = [graph for _, graph, _ in all_processed]
all_labels = [label for _, _, label in all_processed]

print(f"\n成功加载并处理的总样本数: {len(all_processed)}")

print("\n步骤4: 使用所有图创建特征提取器...")
structure_vectorizer, attribute_vectorizer = load_or_create_vectorizers(all_graphs)
# ####简化图
print("\n步骤5: 将所有seq转换为PyG Data格式...")
all_pyg_data = []
all_seq_to_data = {}  # seq -> PyG Data对象
for seq, graph, label in tqdm(all_processed, desc="转换所有图为PyG格式"):
    data = graph_to_pyg_data_dual(graph, label, structure_vectorizer, attribute_vectorizer)
    if data.x.shape[0] > 0:
        all_pyg_data.append((seq, data))
        all_seq_to_data[seq] = data

print(f"有效样本数: {len(all_pyg_data)}")
# 获取训练集、测试集
print("\n步骤6: 按照训练集和测试集划分数据...")
train_seqs_set = set(train_seqs)
test_seqs_set = set(test_seqs)

train_pyg_data = []
test_pyg_data = []
train_seqs_success = []
test_seqs_success = []
test_original_graphs = []

for seq, data in all_pyg_data:
    if seq in train_seqs_set:
        train_pyg_data.append(data)
        train_seqs_success.append(seq)
    elif seq in test_seqs_set:
        test_pyg_data.append(data)
        test_seqs_success.append(seq)
        # 保存原始图用于解释
        for s, g, l in all_processed:
            if s == seq:
                test_original_graphs.append(g)
                break

print(f"训练集有效样本: {len(train_pyg_data)}")
print(f"测试集有效样本: {len(test_pyg_data)}")

# 8. 创建seq到数据的映射（用于解释）
seq_to_data = {seq: data for seq, data in all_pyg_data}
# 训练两类模型，得到决策结果A（均值）
train_loader = DataLoader(train_pyg_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_pyg_data, batch_size=16, shuffle=False)
input_dim_structure = 1
input_dim_attribute = 32 * 7  # 224维

model_structure = StructureGCN(
    structure_in_dim=input_dim_structure,
    hidden_dim=args.hidden_dim,
    out_dim=2,
    dropout=0.3
)

model_attribute = AttributeMLP(
    attribute_in_dim=input_dim_attribute,
    hidden_dim=args.hidden_dim,
    out_dim=2,
    dropout=0.3
)

print(f"\n模型参数统计:")
print(f"  - 图结构模型参数: {sum(p.numel() for p in model_structure.parameters())}")
print(f"  - 节点属性模型参数: {sum(p.numel() for p in model_attribute.parameters())}")

# 训练模型1（图结构）
print(f"\n{'='*60}")
print("训练模型1: 图结构GCN")
print(f"{'='*60}")
train_model(model_structure, train_loader, epochs=args.epochs, lr=args.lr,
            device=args.device, model_name="图结构模型")

# 训练模型2（节点属性）
print(f"\n{'='*60}")
print("训练模型2: 节点属性MLP")
print(f"{'='*60}")
train_model(model_attribute, train_loader, epochs=int(args.epochs*0.25), lr=args.lr,
            device=args.device, model_name="节点属性模型")
y_true, y_pred_model1, y_pred_model2, y_pred_ensemble = ensemble_predict(
    model_structure, model_attribute, test_loader, device=args.device
)

# 获取xml配置信息B

engine = TwoPhaseReasoningEngine(llm_name=llm_name)
seq_xml={}
for seq in test_seqs:
    xml_info = engine.get_theme(seq)
    seq_xml[seq]=xml_info
# 获取关键特征C、关键节点D
features={}
methods={}
features = explain_multiple_seqs(
    seqs=test_seqs,
    model=model_attribute,
    attribute_vectorizer=attribute_vectorizer,
    seq_to_data=seq_to_data,
    device=args.device
)


dual_explainer = DualModelExplainer(model_structure, model_attribute, device=args.device)
dual_explainer.initialize_explainers(test_pyg_data[0])
for target_seq in test_seqs:
    common_nodes = get_common_nodes_by_seq(target_seq, dual_explainer, seq_to_data)
    nodes_tmp=[]
    for one in common_nodes:
        node=seq_to_data[target_seq].node_names[one]
        nodes_tmp.append(node)
    RMs_java, _, _, _ = analyze_risk_components(target_seq, nodes_tmp, jadx_path, smali_path)
    methods[target_seq]=RMs_java

# 并行调用LLM驱动得到测试集三类评估结果
view_train_1='view_train_1'
view_train_2='view_train_2'
view_train_3='view_train_3'
view_test_1='view_test_1'
view_test_2='view_test_2'
view_test_3='view_test_3'
label_feature,label_code,label_xml=invoke_llm_batch(llm_name,y_pred_ensemble,features,methods,seq_xml)

# 训练集生成A，B，C，D
# 并行调用LLM驱动得到训练集三类评估结果
# 训练Dessk矫正模型
# 实现矫正
# 输出结果


