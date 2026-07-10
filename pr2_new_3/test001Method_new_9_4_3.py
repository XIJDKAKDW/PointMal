#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：malwareTest
@File    ：test001Method.py
@IDE     ：PyCharm
@Author  ：常晓松
@Date    ：2025/9/5 9:45
'''
import concurrent
import difflib
import hashlib
import inspect
import itertools
import os
import pickle
import platform
import random
import threading
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import traceback

import joblib
import numpy as np
import pymysql
import torch
import torch.nn as nn
import torch.nn.functional as F  # This is the missing import
from androguard.misc import AnalyzeAPK
from lxml import etree
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer as TF
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import torch
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pr2 import BasicBlockAttrBuilder
from pr2_new_3.GetMultipleMetrixMethod_3 import get_sensitive_apis_extend, generate_apk_method_graph, load_graph, save_graph, \
    list_2_str, revert_smali_batch, \
    get_risk_components, get_sequeces, extract_function_name
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者 ['Microsoft YaHei']、['KaiTi']
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
system = platform.system()

# ==================== 在文件开头添加导入 ====================
import os
import pickle
import numpy as np
import re
from collections import defaultdict
import gensim.downloader as api
from typing import Dict, List, Tuple, Optional, Set
import torch
import torch.nn.functional as F
import json
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
import os
import pickle

# 导入exp_2中的类
from exp_2 import DeepSeekEncoder, ContrastiveModelInference, OriginalContrastiveModel, Config

deepseek_embedding_lock = threading.Lock()
class GroupTestingNodeSelector:
    def __init__(self, seq=None, drebin_model=None, feature_vectorizer=None,
                 contrastive_model_path='visualizations/contrastive_model.pth',
                 deepseek_model_path='/home/changxiaosong/python/malwareTest/deepseek-coder-1.3b-base'):
        self.seq = seq
        self.drebin_model = drebin_model
        self.feature_vectorizer = feature_vectorizer
        self.graph = None  # 保存当前图

        # 存储良性样本和恶意样本的对比学习特征
        self.benign_contrastive_features = []  # 良性样本特征
        self.malicious_contrastive_features = []  # 恶意样本特征

        # 存储各阶段结果
        self.class_risk_scores = {}  # 类风险分数（综合SHAP和恶意性）
        self.method_shap_values = {}
        self.method_malice_scores = {}
        self.method_priority_scores = {}
        self.selected_nodes = []
        self.feature_shap_map = {}

        # 初始化DeepSeek编码器
        self.deepseek_encoder = None#DeepSeekEncoder(deepseek_model_path)
        print(f"[组测试] DeepSeek编码器初始化完成")

        # 加载对比学习模型
        self.contrastive_model = self._load_contrastive_model(contrastive_model_path)

        # 从测试集加载样本特征
        self._load_test_samples()

    def _load_contrastive_model(self, model_path):
        """加载对比学习模型"""
        if not os.path.exists(model_path):
            print(f"[组测试] 警告: 对比学习模型文件 {model_path} 不存在")
            return None

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载原始模型权重
        original_model = OriginalContrastiveModel(
            input_dim=Config.INPUT_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            output_dim=Config.OUTPUT_DIM
        )

        state_dict = torch.load(model_path, map_location='cpu')
        original_model.load_state_dict(state_dict)

        # 创建推理模型
        inference_model = ContrastiveModelInference(
            input_dim=Config.INPUT_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            output_dim=Config.OUTPUT_DIM
        )

        # 复制权重
        with torch.no_grad():
            inference_model.encoder[0].weight.copy_(original_model.encoder[0].weight)
            inference_model.encoder[0].bias.copy_(original_model.encoder[0].bias)
            inference_model.encoder[4].weight.copy_(original_model.encoder[4].weight)
            inference_model.encoder[4].bias.copy_(original_model.encoder[4].bias)
            inference_model.encoder[8].weight.copy_(original_model.encoder[8].weight)
            inference_model.encoder[8].bias.copy_(original_model.encoder[8].bias)

        inference_model.eval()
        inference_model.to(device)
        print(f"[组测试] 对比学习模型加载成功")
        return inference_model

    def _load_test_samples(self):
        """从测试集加载样本特征（与exp_2一致）"""
        try:
            # 加载测试集嵌入和标签
            test_embeddings = np.load(Config.TEST_EMBEDDINGS_PATH)
            test_labels = np.load(Config.TEST_LABELS_PATH)

            # 分离善意和恶意样本
            for i, label in enumerate(test_labels):
                if label == 0:  # 良性
                    self.benign_contrastive_features.append(test_embeddings[i])
                else:  # 恶意
                    self.malicious_contrastive_features.append(test_embeddings[i])

            print(f"[组测试] 已加载 {len(self.benign_contrastive_features)} 个善意样本特征，"
                  f"{len(self.malicious_contrastive_features)} 个恶意样本特征")
        except Exception as e:
            print(f"[组测试] 加载测试样本失败: {e}")

    def reset_state(self, seq):
        """重置状态并设置新的seq"""
        self.seq = seq
        self.class_risk_scores = {}
        self.method_shap_values = {}
        self.method_malice_scores = {}
        self.method_priority_scores = {}
        self.selected_nodes = []
        self.feature_shap_map = {}
        self.graph = None

    def _encode_to_contrastive_space(self, code_text):
        """使用DeepSeek+对比学习模型编码"""
        # DeepSeek编码
        return 0
        # raw_feature = self.deepseek_encoder.encode(code_text)
        #
        # # 对比学习编码
        # if self.contrastive_model:
        #     device = next(self.contrastive_model.parameters()).device
        #     feature_tensor = torch.FloatTensor(raw_feature).unsqueeze(0).to(device)
        #
        #     with torch.no_grad():
        #         # BatchNorm需要batch size>1，复制3次
        #         batch = feature_tensor.repeat(3, 1)
        #         output_batch = self.contrastive_model(batch)
        #         embedding = output_batch[0].cpu().numpy()
        #     return embedding
        # return raw_feature

    def _compute_malice_score(self, node):
        """
        使用边界距离计算节点的恶意性分数（0-1，越高越恶意）
        参考exp_2的BoundaryDistanceAnalyzer
        """
        if self.contrastive_model is None or len(self.benign_contrastive_features) == 0 or len(self.malicious_contrastive_features) == 0:
            return 0.3

        try:
            # 编码节点
            with deepseek_embedding_lock:
                node_embedding = self._encode_to_contrastive_space(node.body)

            # 转换为numpy数组
            benign_features = np.array(self.benign_contrastive_features)
            malicious_features = np.array(self.malicious_contrastive_features)

            # 计算到所有样本的距离
            benign_dists = np.linalg.norm(benign_features - node_embedding, axis=1)
            malicious_dists = np.linalg.norm(malicious_features - node_embedding, axis=1)

            # 计算统计量
            min_benign_dist = np.min(benign_dists)
            min_malicious_dist = np.min(malicious_dists)
            avg_benign_dist = np.mean(benign_dists)
            avg_malicious_dist = np.mean(malicious_dists)

            # 计算边界阈值（90%分位数）
            benign_threshold = np.percentile(benign_dists, 90)
            malicious_threshold = np.percentile(malicious_dists, 90)

            # 判断是否在簇内
            in_benign = min_benign_dist <= benign_threshold
            in_malicious = min_malicious_dist <= malicious_threshold

            # 计算簇内深度
            benign_depth = max(0, benign_threshold - min_benign_dist) / (benign_threshold + 1e-8)
            malicious_depth = max(0, malicious_threshold - min_malicious_dist) / (malicious_threshold + 1e-8)

            # 计算恶意性分数
            if in_malicious and not in_benign:
                score = 0.5 + 0.5 * malicious_depth
            elif in_benign and not in_malicious:
                score = 0.5 - 0.5 * benign_depth
            elif in_malicious and in_benign:
                if malicious_depth > benign_depth:
                    score = 0.5 + 0.25 * (malicious_depth - benign_depth)
                else:
                    score = 0.5 - 0.25 * (benign_depth - malicious_depth)
            else:
                total_dist = avg_benign_dist + avg_malicious_dist + 1e-8
                score = avg_benign_dist / total_dist

            return float(np.clip(score, 0.0, 1.0))

        except Exception as e:
            print(f"[组测试] 计算恶意性分数失败: {e}")
            return 0.3

    def _extract_class_name_from_node(self, node):
        """从节点中提取类名"""
        if hasattr(node, 'name'):
            node_name = node.name
        else:
            node_name = str(node)

        if '->' in node_name:
            class_part = node_name.split('->')[0]
            if class_part.startswith('L') and class_part.endswith(';'):
                return class_part[1:-1].replace('/', '.')
            return class_part
        elif node_name.startswith('L') and node_name.endswith(';'):
            return node_name[1:-1].replace('/', '.')
        else:
            return node_name

    def _extract_class_features(self, graph):
        """提取每个类的所有特征"""
        class_features = {}
        for node in graph.nodes():
            if node.name == 'apk-base':
                continue
            class_name = self._extract_class_name_from_node(node)
            if not class_name:
                continue
            if class_name not in class_features:
                class_features[class_name] = set()
            features = class_features[class_name]

            for perm in node.permission:
                if perm:
                    features.add(f"usedpermissionslist_{perm}".lower())
                    features.add(f"requestedpermissionlist_{perm}".lower())
            for api in node.sensitiveApi:
                if api:
                    features.add(f"restrictedapilist_{api}".lower())
            for api in node.suspiciousApi:
                if api:
                    features.add(f"suspiciousapilist_{api}".lower())
            for url in node.url:
                if url:
                    features.add(f"urldomainlist_{url}".lower())
            if node.component:
                if 'Activity' in node.component or 'activity' in node.component:
                    features.add(f"activitylist_{class_name}".lower())
                elif 'Service' in node.component or 'service' in node.component:
                    features.add(f"servicelist_{class_name}".lower())
                elif 'Receiver' in node.component or 'receiver' in node.component:
                    features.add(f"broadcastreceiverlist_{class_name}".lower())
                elif 'Provider' in node.component or 'provider' in node.component:
                    features.add(f"contentproviderlist_{class_name}".lower())
            for token in node.filterToken:
                if token:
                    features.add(f"intentfilterlist_{token}".lower())
            for hw in node.hardware_component:
                if hw:
                    features.add(f"hardwarecomponentslist_{hw}".lower())
            for provider in node.provider:
                if provider:
                    features.add(f"contentproviderlist_{provider}".lower())
        return class_features

    def _compute_class_risk_score(self, class_name, class_shap, graph):
        return class_shap
        # """
        # 计算类的综合风险分数
        # 结合SHAP值和类内方法的平均恶意性
        # """
        # # 收集该类下所有方法的恶意性分数
        # method_malice_scores = []
        # for node in graph.nodes():
        #     if node.name == 'apk-base':
        #         continue
        #     if self._extract_class_name_from_node(node) == class_name:
        #         # 计算该方法的恶意性
        #         malice = self._compute_malice_score(node)
        #         method_malice_scores.append(malice)
        #
        # if not method_malice_scores:
        #     return class_shap  # 没有方法，只使用SHAP
        #
        # # 综合风险 = SHAP值 * (1 + 平均恶意性)
        # # 这样恶意性高的类即使SHAP稍低也能被选中
        # max_malice = max(method_malice_scores)
        # risk_score = max_malice#class_shap * (1 + avg_malice)
        #
        # return risk_score

    def _compute_class_shap_values(self, graph, explainer, feature_names, x_sample_np):
        """第一阶段：类级风险筛选（结合SHAP和恶意性）"""
        self.graph = graph  # 保存graph供后续使用
        class_features = self._extract_class_features(graph)
        non_zero_indices = np.where(x_sample_np[0] > 0)[0]
        self.feature_shap_map = {}
        # 计算特征SHAP值
        for idx in non_zero_indices:
            feature_name = feature_names[idx]
            shap_value = 0
            if hasattr(explainer, 'shap_values'):
                shap_vals = explainer.shap_values(x_sample_np)
                if isinstance(shap_vals, list):
                    shap_value = shap_vals[1][0, idx]
                else:
                    shap_value = shap_vals[0, idx]
            self.feature_shap_map[feature_name] = shap_value
        # 计算每个类的SHAP值
        class_shap={}
        for class_name, features in class_features.items():
            max_shap = 0.0
            for feature in features:
                if feature in self.feature_shap_map:
                    shap_val = abs(self.feature_shap_map[feature])
                    if shap_val > max_shap:
                        max_shap = shap_val
            class_shap[class_name]=max_shap
        values = np.array(list(class_shap.values()))
        threshold = np.percentile(values, 90)
        for class_name, features in class_features.items():
            max_shap=class_shap[class_name]
            risk_score=max_shap
            if max_shap>threshold:
                # 计算综合风险分数
                risk_score = self._compute_class_risk_score(class_name, max_shap, graph)
            self.class_risk_scores[class_name] = risk_score
        return self.class_risk_scores

    def _select_top_classes(self, top_ratio=0.02):
        """基于风险分数选取top类"""
        if not self.class_risk_scores:
            return []
        # 使用风险分数排序，而不是纯SHAP值
        sorted_classes = sorted(self.class_risk_scores.items(), key=lambda x: x[1], reverse=True)
        k = max(1, int(len(sorted_classes) * top_ratio))
        return [cls for cls, _ in sorted_classes[:k]]

    def _compute_method_shap_allocation(self, graph, top_classes):
        """第二阶段：方法级恶意性定位"""
        method_shap = {}
        if not hasattr(self, 'feature_shap_map') or not self.feature_shap_map:
            return method_shap

        for node in graph.nodes():
            if node.name == 'apk-base':
                continue
            class_name = self._extract_class_name_from_node(node)
            if class_name not in top_classes:
                continue

            node_shap_sum = 0.0
            for perm in node.permission:
                if perm:
                    node_shap_sum += abs(self.feature_shap_map.get(f"usedpermissionslist_{perm}".lower(), 0))
                    node_shap_sum += abs(self.feature_shap_map.get(f"requestedpermissionlist_{perm}".lower(), 0))
            for api in node.sensitiveApi:
                if api:
                    node_shap_sum += abs(self.feature_shap_map.get(f"restrictedapilist_{api}".lower(), 0))
            for api in node.suspiciousApi:
                if api:
                    node_shap_sum += abs(self.feature_shap_map.get(f"suspiciousapilist_{api}".lower(), 0))
            for url in node.url:
                if url:
                    node_shap_sum += abs(self.feature_shap_map.get(f"urldomainlist_{url}".lower(), 0))

            if node_shap_sum == 0:
                continue

            class_risk = self.class_risk_scores.get(class_name, 0)
            class_features = self._extract_class_features(graph).get(class_name, set())
            class_shap_sum = 0.0
            for feat in class_features:
                class_shap_sum += abs(self.feature_shap_map.get(feat, 0))

            weight = node_shap_sum / max(class_shap_sum, 1e-10)
            method_shap[node] = class_risk * weight  # 使用class_risk替代class_shap

        return method_shap

    def _compute_priority_scores(self, method_shap_values, malice_scores):
        """
        综合优先级评分（基于双排名）
        优先级分数 = 基于SHAP值和恶意性分数的双排名综合计算
        两类排名都靠前的节点分数高，有一个排名靠后的节点分数低
        """
        priority_scores = {}
        epsilon = 1e-8

        if not method_shap_values:
            return priority_scores

        # 获取所有节点列表
        nodes = list(method_shap_values.keys())

        # 1. 计算SHAP值排名（基于绝对值，值越大排名越靠前）
        shap_abs_values = {node: abs(method_shap_values[node]) for node in nodes}
        sorted_by_shap = sorted(shap_abs_values.items(), key=lambda x: x[1], reverse=True)
        shap_rank = {node: idx + 1 for idx, (node, _) in enumerate(sorted_by_shap)}  # 排名从1开始

        # 2. 计算恶意性分数排名（值越大排名越靠前）
        # 使用传入的malice_scores，如果节点不存在则使用默认值0.3
        malice_values = {node: malice_scores.get(node, 0.3) for node in nodes}
        sorted_by_malice = sorted(malice_values.items(), key=lambda x: x[1], reverse=True)
        malice_rank = {node: idx + 1 for idx, (node, _) in enumerate(sorted_by_malice)}  # 排名从1开始

        # 3. 计算节点总数
        n = len(nodes)

        # 4. 基于双排名计算综合优先级分数
        for node in nodes:
            # 获取两个排名
            s_rank = shap_rank[node]
            m_rank = malice_rank[node]

            # 方法1：使用乘积（两个排名都靠前时分数高）
            # 将排名转换为0-1之间的分数（排名越靠前分数越高）
            norm_shap_score = (n - s_rank + 1) / n  # 排名1: 1.0, 排名n: 1/n
            norm_malice_score = (n - m_rank + 1) / n

            # 综合分数 = 两个分数的乘积（两者都大时乘积才大）
            combined_score = norm_shap_score * norm_malice_score

            priority_scores[node] = combined_score

        # 可选：对最终分数进行归一化到0-1范围
        max_score = max(priority_scores.values()) if priority_scores else 1.0
        if max_score > 0:
            for node in priority_scores:
                priority_scores[node] = priority_scores[node] / max_score

        return priority_scores
    def _adaptive_threshold_selection(self, priority_scores):
        """第四阶段：自适应代表节点选择（手肘法）"""
        if not priority_scores:
            return []

        sorted_nodes = sorted(priority_scores.items(), key=lambda x: x[1], reverse=True)
        scores = [score for _, score in sorted_nodes]

        if len(scores) <= 2:
            k = min(1, len(scores))
            return [node for node, _ in sorted_nodes[:k]]

        # 计算二阶导数找拐点
        second_derivatives = []
        for i in range(1, len(scores) - 1):
            d2 = scores[i+1] + scores[i-1] - 2 * scores[i]
            second_derivatives.append((i, abs(d2)))

        if second_derivatives:
            elbow_idx, _ = max(second_derivatives, key=lambda x: x[1])
            k = elbow_idx + 1
        else:
            k = min(3, len(scores))

        return [node for node, _ in sorted_nodes[:k]]

    def select_suspicious_nodes(self, graph, explainer, feature_names, x_sample_np):
        """完整的组测试节点选择流程（改进版：类级筛选使用风险分数）"""
        # print(f"[组测试] 开始对序列 {self.seq} 进行分层恶意性评估...")

        # 第一阶段：类级风险筛选（综合SHAP和恶意性）
        # print("[阶段1] 计算类级风险分数（SHAP + 恶意性）...")
        self._compute_class_shap_values(graph, explainer, feature_names, x_sample_np)
        top_classes = self._select_top_classes(top_ratio=0.02)
        # print(f"        - 高风险类数量: {len(top_classes)}")
        # 第二阶段：方法级恶意性定位
        # print("[阶段2] 分配方法级SHAP值并计算恶意性分数...")
        self.method_shap_values = self._compute_method_shap_allocation(graph, top_classes)
        # 计算每个方法的恶意性分数（基于对比学习边界距离）
        method_count = 0
        for node in self.method_shap_values.keys():
            self.method_malice_scores[node] = self._compute_malice_score(node)
            method_count += 1

        # if method_count > 0:
        #     scores = list(self.method_malice_scores.values())
        #     print(f"        - 待评估方法数量: {method_count}")
        #     print(f"        - 恶意性分数范围: {min(scores):.3f} ~ {max(scores):.3f}")
        #     print(f"        - 平均恶意性: {np.mean(scores):.3f}")

        # 第三阶段：综合优先级排序
        # print("[阶段3] 计算综合优先级分数...")
        self.method_priority_scores = self._compute_priority_scores(
            self.method_shap_values, self.method_malice_scores
        )

        # 第四阶段：自适应代表节点选择
        # print("[阶段4] 自适应阈值选择代表节点...")
        self.selected_nodes = self._adaptive_threshold_selection(self.method_priority_scores)
        # print(f"        - 最终选择代表节点数量: {len(self.selected_nodes)}")
        return self.selected_nodes

    def visualize_selected_nodes(self, output_path=None):
        """可视化已选择的代表节点在对比学习空间中的位置（t-SNE）"""
        if not self.selected_nodes:
            print("[组测试] 没有已选择的代表节点，无法可视化")
            return None

        # 收集所有节点的嵌入
        all_embeddings = []
        all_labels = []
        all_names = []

        # 添加良性样本
        for i, feat in enumerate(self.benign_contrastive_features):
            all_embeddings.append(feat)
            all_labels.append(0)  # 良性
            all_names.append(f"benign_{i}")

        # 添加恶意样本
        for i, feat in enumerate(self.malicious_contrastive_features):
            all_embeddings.append(feat)
            all_labels.append(1)  # 恶意
            all_names.append(f"malicious_{i}")

        # 添加选中的节点
        for i, node in enumerate(self.selected_nodes):
            node_embedding = self._encode_to_contrastive_space(node.body)
            all_embeddings.append(node_embedding)
            all_labels.append(2)  # 选中节点
            all_names.append(f"selected_{i}")

        # t-SNE降维
        all_embeddings = np.array(all_embeddings)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)-1))
        reduced = tsne.fit_transform(all_embeddings)

        # 分离数据
        n_benign = len(self.benign_contrastive_features)
        n_malicious = len(self.malicious_contrastive_features)
        n_selected = len(self.selected_nodes)

        benign_red = reduced[:n_benign]
        malicious_red = reduced[n_benign:n_benign+n_malicious]
        selected_red = reduced[n_benign+n_malicious:]

        # 绘图
        plt.figure(figsize=(12, 8))
        plt.scatter(benign_red[:, 0], benign_red[:, 1], c='blue', marker='o',
                    label=f'Benign ({n_benign})', alpha=0.6, s=30)
        plt.scatter(malicious_red[:, 0], malicious_red[:, 1], c='red', marker='^',
                    label=f'Malicious ({n_malicious})', alpha=0.6, s=30)
        plt.scatter(selected_red[:, 0], selected_red[:, 1], c='green', marker='D',
                    label=f'Selected ({n_selected})', alpha=0.9, s=100, edgecolors='black', linewidth=1.5)

        plt.title(f'Selected Nodes in Contrastive Space (t-SNE) - Seq {self.seq}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            plt.show()
            return None




def visualize_suspicious_nodes_in_contrastive_space(group_selector, suspicious_nodes, output_path=None):
    """使用PCA可视化可疑节点与善意样本、恶意样本和模糊样本的相对距离"""
    if not group_selector.contrastive_model or len(group_selector.benign_contrastive_features) == 0 or len(group_selector.malicious_contrastive_features) == 0:
        print("[可视化] 警告: 模型或样本特征未加载")
        return None

    try:
        # 1. 提取特征
        suspicious_features, suspicious_names, suspicious_scores = [], [], []

        # 获取每个节点的benign_similarities分数
        for i, node in enumerate(suspicious_nodes):
            features = group_selector._extract_node_feature_vector(node.body)
            if features is not None:
                suspicious_features.append(group_selector._encode_to_contrastive_space(features))

                # 获取节点名称用于显示
                name = getattr(node, 'name', f"Node_{i}")
                if '->' in name:
                    name = name.split('->')[-1].split('(')[0]
                suspicious_names.append(f"{i+1}")  # 序号从1开始

                # 获取该节点的benign_similarities分数
                score = group_selector.method_benign_similarities.get(node, 0.5)
                suspicious_scores.append(score)

        # 2. 准备数据
        benign_array = np.array(group_selector.benign_contrastive_features)
        malicious_array = np.array(group_selector.malicious_contrastive_features)
        ambiguous_array = np.array(group_selector.ambiguious_contrastive_features)
        suspicious_array = np.array(suspicious_features)

        if len(benign_array) == 0 or len(malicious_array) == 0:
            return None
        # 训练PCA模型
        # pca = joblib.load(group_selector.pca)
        # print("PCA模型已加载")


        # 使用训练好的PCA模型转换所有数据
        benign_2d = benign_array#pca.transform(benign_array)
        malicious_2d = malicious_array#pca.transform(malicious_array)
        ambiguous_2d = ambiguous_array#pca.transform(ambiguous_array)
        suspicious_2d = suspicious_array#pca.transform(suspicious_array)

        # 计算方差解释比例

        # 4. 可视化
        plt.figure(figsize=(16, 12))

        # 绘制良性样本
        if len(benign_2d):
            plt.scatter(benign_2d[:, 0], benign_2d[:, 1], c='green', alpha=0.15, s=20,
                        label=f'Benign ({len(benign_array)})', edgecolors='none')

        # 绘制恶意样本
        if len(malicious_2d):
            plt.scatter(malicious_2d[:, 0], malicious_2d[:, 1], c='red', alpha=0.15, s=20,
                        label=f'Malicious ({len(malicious_array)})', edgecolors='none')

        # 绘制模糊样本
        if len(ambiguous_2d):
            plt.scatter(ambiguous_2d[:, 0], ambiguous_2d[:, 1], c='blue', alpha=0.15, s=20,
                        label=f'Ambiguous ({len(ambiguous_array)})', edgecolors='none')

        if len(suspicious_2d) > 0:
            # 使用benign_similarities分数作为颜色映射（分数越高越可疑）
            scatter = plt.scatter(suspicious_2d[:, 0], suspicious_2d[:, 1],
                                  c=np.array(suspicious_scores),
                                  cmap='YlOrRd', s=80,
                                  edgecolors='black', linewidth=1.5,
                                  vmin=0, vmax=1,
                                  label=f'Suspicious ({len(suspicious_2d)})')

            # 添加颜色条
            cbar = plt.colorbar(scatter)
            cbar.set_label('Suspicious Score (higher = more malicious)', fontsize=10)

            # 为每个可疑节点添加序号标签
            for i, (x, y) in enumerate(suspicious_2d):
                plt.annotate(suspicious_names[i], (x, y),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=8, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))

        # 更新标题，添加PCA的方差解释信息
        plt.xlabel(f'Principal Component 1', fontsize=11)
        plt.ylabel(f'Principal Component 2', fontsize=11)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(alpha=0.3)

        # 保存
        if output_path is None:
            import pandas as pd
            output_path = f"suspicious_nodes_pca_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

        print(f"[可视化] 已保存: {output_path}")
        print(f"[可视化] 样本数量: 良性={len(benign_array)}, 恶意={len(malicious_array)}, 模糊={len(ambiguous_array)}, 可疑={len(suspicious_array)}")

        # 打印节点序号与名称的对应关系
        print("\n[可视化] 节点序号对应关系和分数：")
        for i, node in enumerate(suspicious_nodes):
            name = getattr(node, 'name', f"Node_{i}")
            score_sim = group_selector.method_benign_similarities.get(node, 0.5)
            score_shap = group_selector.method_shap_values.get(node, 0.5)
            print(f"  {i+1:2d}. {name} (similar score: {score_sim:.3f}\t(SHAP: {score_shap:.3f})")

        return output_path

    except Exception as e:
        print(f"[可视化] 失败: {e}")
        import traceback
        traceback.print_exc()
        return None
class ApkAnalyzer:
    def __init__(self, apk_path, smali_path):
        self.apk_path = apk_path
        self.smali_path = smali_path
        self.AndroidManifest = smali_path + os.sep + "AndroidManifest.xml"

        # 确保AndroidManifest.xml存在
        if not Path(self.AndroidManifest).exists():
            try:
                a, d, dx = AnalyzeAPK(apk_path)
                manifest_element = a.xml["AndroidManifest.xml"]
                directory = os.path.dirname(self.AndroidManifest)
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                # 方法2: 创建完整的XML文档
                with open(self.AndroidManifest, "w", encoding='utf-8') as f:
                    # 创建完整的XML文档结构
                    xml_declaration = '<?xml version="1.0" encoding="utf-8"?>\n'

                    # 获取manifest内容
                    manifest_content = etree.tostring(
                        manifest_element,
                        pretty_print=True,
                        encoding='utf-8',
                        xml_declaration=False
                    ).decode('utf-8')

                    f.write(xml_declaration + manifest_content)

            except Exception as e:
                print(f"生成AndroidManifest.xml失败: {e}")

        # 初始化receiver-action映射
        self.receiver_actions = defaultdict(list)
        self._parse_android_manifest()
        self.provider_names=self.get_provider_names()
        self.hardware_component=self.get_hardware_component()


    def _parse_android_manifest(self):
        """解析AndroidManifest.xml，提取receiver和action的对应关系"""
        try:
            # 读取XML文件
            with open(self.AndroidManifest, 'r', encoding='utf-8') as f:
                xml_content = f.read()

            # 解析XML内容
            root = ET.fromstring(xml_content)

            # 定义命名空间
            namespace = {'android': 'http://schemas.android.com/apk/res/android'}

            # 查找application元素
            application = root.find('application')
            if application is None:
                return

            # 查找所有的receiver元素
            for receiver in application.findall('receiver'):
                receiver_name = receiver.get('{http://schemas.android.com/apk/res/android}name')
                if receiver_name is None:
                    continue

                # 查找receiver下的intent-filter
                for intent_filter in receiver.findall('intent-filter'):
                    # 查找action元素
                    for action in intent_filter.findall('action'):
                        action_name = action.get('{http://schemas.android.com/apk/res/android}name')
                        if action_name:
                            self.receiver_actions[receiver_name].append(action_name)

                # 如果没有找到action，添加空列表
                if receiver_name not in self.receiver_actions:
                    self.receiver_actions[receiver_name] = []

        except ET.ParseError as e:
            print(f"XML解析错误: {e}")
        except Exception as e:
            print(f"解析AndroidManifest.xml时发生错误: {e}")

    def search_actions_by_classname(self, classname):
        # 处理L格式的类名（如：Lmm/sms/purchasesdk/sms/SMSReceiver;）
        if '->' in classname:
            classname=classname.split('->')[0]
        if classname.startswith('L') and classname.endswith(';'):
            # 转换为标准Java类名格式
            classname = classname[1:-1].replace('/', '.')

        # 直接搜索
        if classname in self.receiver_actions:
            return self.receiver_actions[classname]
        # 如果没有直接匹配，尝试模糊搜索（处理可能的包名简写等情况）
        for receiver_name, actions in self.receiver_actions.items():
            if receiver_name.endswith('.' + classname) or receiver_name == classname:
                return actions

        return []


    def get_provider_names(self):
        provider_names = []
        try:
            # 读取XML文件
            with open(self.AndroidManifest, 'r', encoding='utf-8') as f:
                xml_content = f.read()

            # 解析XML内容
            root = ET.fromstring(xml_content)

            # 定义命名空间
            namespace = {'android': 'http://schemas.android.com/apk/res/android'}

            # 查找application元素
            application = root.find('application')
            if application is None:
                return provider_names

            # 查找所有的provider元素
            for provider in application.findall('provider'):
                provider_name = provider.get('{http://schemas.android.com/apk/res/android}name')
                if provider_name:
                    provider_names.append(provider_name)

        except ET.ParseError as e:
            print(f"XML解析错误: {e}")
        except Exception as e:
            print(f"解析AndroidManifest.xml时发生错误: {e}")

        return provider_names
    def get_hardware_component(self):
        """解析AndroidManifest.xml，提取uses-feature的android:name内容"""
        hardware_features = []
        try:
            # 读取XML文件
            with open(self.AndroidManifest, 'r', encoding='utf-8') as f:
                xml_content = f.read()

            # 解析XML内容
            root = ET.fromstring(xml_content)

            # 查找所有的uses-feature元素
            for uses_feature in root.findall('uses-feature'):
                feature_name = uses_feature.get('{http://schemas.android.com/apk/res/android}name')
                if feature_name:
                    hardware_features.append(feature_name)

        except ET.ParseError as e:
            print(f"XML解析错误: {e}")
        except Exception as e:
            print(f"解析AndroidManifest.xml时发生错误: {e}")

        return hardware_features
    def get_uses_permissions(self):
        """
        获取AndroidManifest.xml中所有uses-permission的android:name值，作为list返回
        """
        permissions = []
        try:
            # 读取XML文件
            with open(self.AndroidManifest, 'r', encoding='utf-8') as f:
                xml_content = f.read()

            # 解析XML内容
            root = ET.fromstring(xml_content)

            # 定义命名空间
            namespace = {'android': 'http://schemas.android.com/apk/res/android'}

            # 查找所有的uses-permission元素
            for uses_permission in root.findall('uses-permission'):
                permission_name = uses_permission.get('{http://schemas.android.com/apk/res/android}name')
                if permission_name:
                    permissions.append(permission_name)

        except ET.ParseError as e:
            print(f"XML解析错误: {e}")
        except Exception as e:
            print(f"解析AndroidManifest.xml时发生错误: {e}")

        return permissions
    def get_all_filter_actions(self):
        """
        获取所有intent-filter中的action android:name值，放入list中返回
        """
        all_actions = []
        try:
            # 读取XML文件
            with open(self.AndroidManifest, 'r', encoding='utf-8') as f:
                xml_content = f.read()

            # 解析XML内容
            root = ET.fromstring(xml_content)

            # 定义命名空间
            namespace = {'android': 'http://schemas.android.com/apk/res/android'}

            # 查找所有包含intent-filter的元素（activity, service, receiver等）
            elements_with_intent_filters = []

            # 查找application元素
            application = root.find('application')
            if application is not None:
                # 查找所有可能包含intent-filter的元素类型
                element_types = ['activity', 'service', 'receiver', 'provider']

                for element_type in element_types:
                    elements_with_intent_filters.extend(application.findall(element_type))

            # 遍历所有可能包含intent-filter的元素
            for element in elements_with_intent_filters:
                # 查找元素下的所有intent-filter
                for intent_filter in element.findall('intent-filter'):
                    # 查找action元素
                    for action in intent_filter.findall('action'):
                        action_name = action.get('{http://schemas.android.com/apk/res/android}name')
                        if action_name and action_name not in all_actions:
                            all_actions.append(action_name)

        except ET.ParseError as e:
            print(f"XML解析错误: {e}")
        except Exception as e:
            print(f"解析AndroidManifest.xml时发生错误: {e}")

        return all_actions

class SmaliAnalyzer:
    def __init__(self, project_root):
        self.project_root = project_root
        self.project_packages = self._extract_project_packages()

    def _extract_project_packages(self):
        """从项目目录中提取所有包名"""
        packages = set()
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                if file.endswith('.smali'):
                    rel_path = os.path.relpath(root, self.project_root)
                    package_name = 'L' + rel_path.replace('/', '/') + '/'
                    packages.add(package_name)
        return list(packages)

    def analyze_smali_file(self, file_path):
        """分析 Smali 文件中的所有 invoke 指令"""
        project_invokes = []
        system_invokes = []

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if line.startswith('invoke-'):
                if self._is_project_invoke(line):
                    project_invokes.append(line)
                else:
                    system_invokes.append(line)

        return project_invokes, system_invokes
    def path_matches(self,pattern, target):
        pattern_norm = pattern.replace('\\', '/')
        target_norm = target.replace('\\', '/')
        return pattern_norm in target_norm
    def _is_project_invoke(self, invoke_line):
        """判断单个 invoke 指令是否调用项目代码"""
        # 提取被调用的类名
        class_pattern = r'invoke-\w+ \{[^}]+\}, (L[^;]+;)->'
        match = re.search(class_pattern, invoke_line)

        if not match:
            return False

        class_name = match.group(1)

        # 系统包过滤
        system_prefixes = [
            'Ljava/', 'Ljavax/', 'Landroid/', 'Ldalvik/',
            'Lorg/xml/', 'Lorg/json/', 'Lorg/w3c/', 'Lorg/apache/'
        ]

        for prefix in system_prefixes:
            if class_name.startswith(prefix):
                return False

        # 项目包匹配
        for prefix in self.project_packages:
            if 'purchasesdk' in prefix:
                pass
            if self.path_matches(prefix, class_name):
                return True

        return False


def string_similarity(s1, s2):
    # 快速检查完全匹配
    if s1 == s2:
        return 1.0
    # 快速检查前缀匹配
    if s1.startswith(s2) or s2.startswith(s1):
        min_len = min(len(s1), len(s2))
        return min_len / max(len(s1), len(s2))
    # 回退到原来的实现
    return difflib.SequenceMatcher(None, s1, s2).ratio()
def get_apk_hash(apk_file):
    """计算APK文件的哈希值"""
    hasher = hashlib.md5()
    with open(apk_file, 'rb') as f:  # 使用with语句自动关闭文件
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def extract_serializable_analysis_data(d, dx):
    """提取可序列化的分析数据"""
    # 这里只提取基本数据，避免保存复杂的分析对象
    return {
        'class_names': [c.get_name() for c in d.get_classes()],
        'method_count': sum(len(c.get_methods()) for c in d.get_classes()),
        'string_resources': list(d.get_strings()),
    }
def string_similarity(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).ratio()
def analyze_apk_with_cache(apk_file):
    """带缓存的APK分析（修复版本）"""
    cache_dir = "/tmp/androguard_cache"
    os.makedirs(cache_dir, exist_ok=True)

    apk_hash = get_apk_hash(apk_file)
    cache_file = os.path.join(cache_dir, f"{apk_hash}.pkl")

    if os.path.exists(cache_file):
        print("Loading from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # 分析APK，但只保存可pickle的数据

    a, d, dx = AnalyzeAPK(apk_file)

    # 提取可序列化的数据
    cache_data = {
        'apk_info': extract_serializable_apk_info(a),
        'analysis_data': extract_serializable_analysis_data(d, dx)
    }

    # 保存到缓存
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)

    return cache_data

def extract_serializable_apk_info(a):
    """提取可序列化的APK信息"""
    return {
        'package_name': a.get_package(),
        'version_name': a.get_version_name(),
        'version_code': a.get_version_code(),
        'permissions': a.get_permissions(),
        'activities': a.get_activities(),
        'services': a.get_services(),
        'receivers': a.get_receivers(),
        'providers': a.get_providers(),
    }
def get_path_from_db(conn,dataset_sql):
    file_path=[]
    label=[]
    cur = conn.cursor()
    result = cur.execute(dataset_sql)
    for i in range(result):
        oneRow = cur.fetchone()
        file_path.append(oneRow[0])
        label.append(oneRow[1])
    cur.close()
    return file_path,label
def plot_value(y_true, y_pred):
    import sklearn as sk
    from sklearn.metrics import balanced_accuracy_score
    balance_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = sk.metrics.precision_score(y_true, y_pred)
    recall = sk.metrics.recall_score(y_true, y_pred)
    f1_value = sk.metrics.f1_score(y_true, y_pred)
    return balance_accuracy, precision, recall, f1_value
def print_message(message,balance_accuracy, precision, recall, f1_value):
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(time.time())))
    print ('时间：' + t+ " ")
    print (message+' '+str(balance_accuracy)+' '+ str(precision)+' '+ str(recall)+' '+str(f1_value))
def get_connection():
    host = "211.65.82.10"
    user = "root"
    password = 'chang123'
    db = 'malware_db'
    conn = pymysql.connect(
        host=host,
        user=user,
        password=password,
        db=db,
        charset='utf8',
        # autocommit=True,    # 如果插入数据，， 是否自动提交? 和conn.commit()功能一致。
    )
    return conn

def get_feature_file_label_by_seq(conn,seqs):
    file_paths,labels=[],[]
    for i in seqs:
        dataset_sql = "select b.path, case  a.label when 'B' then 0 else 1 end " \
                      "from app_label a,drebin_feature b " \
                      "where  a.seq=b.apkSeq and  a.seq="+str(i);
        file_paths_tmp,labels_tmp= get_path_from_db(conn,dataset_sql)
        file_paths.extend(file_paths_tmp)
        labels.extend(labels_tmp)
    return file_paths,labels
def get_apk_file(conn,seqs):
    file_paths,labels=[],[]
    for i in seqs:
        dataset_sql = "select a.path, case  a.label when 'B' then 0 else 1 end " \
                      "from app_label a,drebin_feature b " \
                      "where  a.seq=b.apkSeq and  a.seq="+str(i);
        file_paths_tmp,labels_tmp= get_path_from_db(conn,dataset_sql)
        file_paths.extend(file_paths_tmp)
        labels.extend(labels_tmp)
    return file_paths,labels

def shuffle_data(paths, labels,seed):
    c = list(zip(paths, labels))  # 将a,b整体作为一个zip,每个元素一一对应后打乱
    random.seed(seed)
    random.shuffle(c)  # 打乱c
    paths[:], labels[:]= zip(*c)  # 将打乱的c解开
    return paths, labels
def get_obj( file_path):
    f = open(file_path)
    seqs=[]
    for line in f.readlines():
        if len(line) <= 0:
            continue
        line=line.replace('\n','')
        seqs.append(line)
    return seqs

class SparseToDense(nn.Module):
    """自定义稀疏转稠密层"""
    def forward(self, x):
        if isinstance(x, torch.sparse.Tensor):
            return x.to_dense()
        return x


class SparseDataset(Dataset):
    def __init__(self, X_sparse, y, y_binary_cat=None, weights=None):
        """
        Args:
            X_sparse (csr_matrix): 稀疏特征矩阵
            y (np.ndarray): 标签（整数格式）
            y_binary_cat (np.ndarray): 二分类的 one-hot 标签（可选）
            weights (np.ndarray): 样本权重（可选）
        """
        # 转换为 PyTorch 稀疏张量（CSR 格式）
        self.X = torch.sparse_csr_tensor(
            crow_indices=torch.tensor(X_sparse.indptr, dtype=torch.int64),
            col_indices=torch.tensor(X_sparse.indices, dtype=torch.int64),
            values=torch.tensor(X_sparse.data, dtype=torch.float32),
            size=X_sparse.shape
        )
        tmp_y={}
        for one in y:
            tmp_y[one]=torch.tensor(y[one], dtype=torch.long)
        self.y = tmp_y  # 整数标签

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        """返回第 idx 个样本的稀疏特征、标签及权重（如果存在）"""
        # 提取稀疏特征（CSR 格式的行切片）
        row_start = self.X.crow_indices()[idx]
        row_end = self.X.crow_indices()[idx + 1]
        col_indices = self.X.col_indices()[row_start:row_end]
        values = self.X.values()[row_start:row_end]

        # 构造稀疏特征（COO 格式，适用于单行）
        sparse_feature = torch.sparse_coo_tensor(
            indices=torch.stack([torch.zeros_like(col_indices), col_indices]),
            values=values,
            size=(1, self.X.size(1)),
            dtype=torch.float32
        ).coalesce()

        # 返回数据（根据是否提供 y_binary_cat 和 weights 调整）
        tmp_y={}
        for one in self.y:
            tmp_y[one]=self.y[one][idx]
        return sparse_feature, tmp_y,

def sparse_collate_fn(batch):
    """
    自定义 collate_fn，用于处理稀疏张量的批量数据
    Args:
        batch: 列表，每个元素是 __getitem__ 的返回值
    Returns:
        批量的稀疏张量、标签及其他可选数据
    """
    # 分离不同部分的数据
    if len(batch[0]) == 4:  # 包含 sparse_feature, y, y_binary_cat, weights
        sparse_features, ys, y_binary_cats, weights = zip(*batch)
        y_binary_cats = torch.stack(y_binary_cats)
        weights = torch.stack(weights)
    elif len(batch[0]) == 3:  # 包含 sparse_feature, y, y_binary_cat 或 weights
        sparse_features, ys, extras = zip(*batch)
        if extras[0].dim() == 1:  # 判断是 weights
            weights = torch.stack(extras)
            y_binary_cats = None
        else:  # 是 y_binary_cat
            y_binary_cats = torch.stack(extras)
            weights = None
    else:  # 只有 sparse_feature 和 y
        sparse_features, ys = zip(*batch)
        ys_new={}
        for i in ys:
            for key in set(i.keys()):
                if key not in ys_new:
                    ys_new[key]=[i[key]]
                else:
                    tmp=ys_new[key]
                    tmp.append(i[key])
                    ys_new[key]=tmp
        for i in ys_new:
            ys_new[i]=torch.stack(ys_new[i])
        ys=ys_new
        y_binary_cats, weights = None, None

    # 合并稀疏张量（沿 batch 维度拼接）
    batch_size = len(sparse_features)
    all_indices = []
    all_values = []
    offset = 0

    for i, sp_tensor in enumerate(sparse_features):
        # 调整 indices 的 batch 维度
        indices = sp_tensor.indices()
        indices[0, :] += i  # 第0维是 batch 维度
        all_indices.append(indices)
        all_values.append(sp_tensor.values())

    # 拼接所有稀疏张量
    if all_indices:
        all_indices = torch.cat(all_indices, dim=1)
        all_values = torch.cat(all_values)
        batch_sparse = torch.sparse_coo_tensor(
            indices=all_indices,
            values=all_values,
            size=(batch_size, sparse_features[0].size(1)),
            dtype=sparse_features[0].dtype
        )
    else:
        batch_sparse = torch.sparse_coo_tensor(size=(batch_size, 0))  # 空矩阵


    # 返回结果
    if y_binary_cats is not None and weights is not None:
        return batch_sparse, ys, y_binary_cats, weights
    elif y_binary_cats is not None:
        return batch_sparse, ys, y_binary_cats
    elif weights is not None:
        return batch_sparse, ys, weights
    else:
        return batch_sparse, ys


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        # Vectorized computation of margin list
        cls_num_list = torch.as_tensor(cls_num_list, dtype=torch.float32)
        m_list = torch.rsqrt(torch.sqrt(cls_num_list))  # More efficient than 1/sqrt(sqrt())
        m_list = (m_list * (max_m / m_list.max())).clamp(min=1e-6)  # Add small epsilon
        self.register_buffer('m_list', m_list)  # Proper buffer registration

        self.s = s
        if weight is not None:
            weight = torch.as_tensor(weight, dtype=torch.float32)
        self.register_buffer('weight', weight)  # Also register weight as buffer

    def forward(self, x, target):
        # Get device from input tensor
        device = x.device

        # Create one-hot style index efficiently
        batch_size = x.size(0)
        target_margins = self.m_list[target].view(-1, 1)

        # Compute adjusted logits
        x_m = x - target_margins.expand_as(x) * (torch.arange(x.size(1), device=device) == target.view(-1, 1)).float()
        return F.cross_entropy(self.s * x_m, target, weight=self.weight)
def get_multi_batch(device,y_labels):
    branch_name = []
    loss_weights = {}
    train_labels = {}
    criterions={}
    train_f=set(y_labels.keys())
    for i, one in enumerate(train_f):
        branch_name.append('branch' + str(i + 1))
        if one == 'begin':
            loss_weights['branch' + str(i + 1) ] = 1.8
        else:
            loss_weights['branch' + str(i + 1) ] = 0.8
        train_labels['branch' + str(i + 1) ] = y_labels[one].to(device)
    for i, one in enumerate(train_f):
        criterions['branch' + str(i + 1)] = nn.BCELoss(weight=torch.tensor(loss_weights['branch' + str(i + 1)]).to(device))
    return branch_name, criterions, train_labels
from sklearn.ensemble import GradientBoostingClassifier
import shap

class Drebin_GBM:
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        self.is_fitted = False

    def forward(self, x):
        """为了保持接口兼容性，保留forward方法"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        # 这个方法在预测时不会被调用，只是为了兼容接口
        return [torch.tensor([[0.0, 0.0]])]  # 返回一个虚拟值

    def fit(self, X, y):
        """训练GBM模型"""
        # 如果X是稀疏矩阵，转换为稠密矩阵
        if hasattr(X, 'toarray'):
            X = X.toarray()
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        if hasattr(X, 'toarray'):
            X = X.toarray()
        return self.model.predict_proba(X)

    def predict(self, X):
        """预测类别"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        if hasattr(X, 'toarray'):
            X = X.toarray()
        return self.model.predict(X)


def train_model(device, model, num_epoch, features_train_new, y_labels, initial_learning_rate, batch_size, patience=5, min_delta=0.001):
    # 对于GBM模型，使用不同的训练方式
    if isinstance(model, Drebin_GBM):
        # 提取标签
        if isinstance(y_labels['final'], np.ndarray):
            if y_labels['final'].ndim == 2 and y_labels['final'].shape[1] == 2:  # One-hot
                y_train = np.argmax(y_labels['final'], axis=1)
            else:  # Already class indices
                y_train = y_labels['final']
        elif isinstance(y_labels['final'], torch.Tensor):
            if y_labels['final'].dim() == 2 and y_labels['final'].size(1) == 2:  # One-hot
                y_train = torch.argmax(y_labels['final'], dim=1).numpy()
            else:
                y_train = y_labels['final'].numpy()
        else:
            y_train = y_labels['final']

        # 训练GBM模型
        model.fit(features_train_new, y_train)
        print("GBM模型训练完成")

def get_feature(seqs_train,seqs_test):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 清理显存
    print("特征生成", flush=True)
    conn = get_connection()

    file_paths_train, labels_train = get_feature_file_label_by_seq(conn, seqs_train)
    file_paths_test, labels_test = get_feature_file_label_by_seq(conn, seqs_test)
    conn.close()

    system = platform.system()

    if not system == "Linux":
        file_paths_train_new=[]
        for one in file_paths_train:
            file_paths_train_new.append(one.replace('/home/changxiaosong/dataset','D:'))
        file_paths_test_new=[]
        for one in file_paths_test:
            file_paths_test_new.append(one.replace('/home/changxiaosong/dataset','D:'))
        file_paths_train=file_paths_train_new
        file_paths_test=file_paths_test_new

    FeatureVectorizer = TF(input="filename", tokenizer=lambda x: x.split('\n'), token_pattern=None,
                           binary=True)
    num_classes = 2  # 类别数量，0 和 1
    # 转为 one-hot 编码
    labels_train = np.eye(num_classes)[labels_train]
    labels_test = np.eye(num_classes)[labels_test]
    features_train = FeatureVectorizer.fit_transform(file_paths_train)
    features_test = FeatureVectorizer.transform(file_paths_test)
    return FeatureVectorizer,features_train,labels_train,features_test,labels_test


class WrappedModel(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model

    def forward(self, x):
        # 确保输入在GPU上
        if isinstance(x, torch.Tensor):
            if x.is_sparse:
                x = x.to_dense()
            # 如果x在GPU上，确保后续操作也在GPU上
            device = x.device
            x_np = x.cpu().detach().numpy()
        else:
            x_np = x
            device = torch.device('cpu')

        # 预测概率并返回logits
        proba = self.original_model.predict_proba(x_np)
        # 将概率转换为logits
        logits = torch.tensor(np.log(proba / (1 - proba + 1e-8)), dtype=torch.float32)

        # 确保logits在正确的设备上
        return logits.to(device)

def get_file_hash(file_path):
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def determine_feature_type(feature_name):
    feature_name_lower = feature_name.lower()

    if 'servicelist_' in feature_name_lower or 'activitylist_' in feature_name_lower \
            or 'contentproviderlist_' in feature_name_lower or 'broadcastreceiverlist_' in feature_name_lower \
            or 'hardwarecomponentslist_' in feature_name_lower:
        return "component"
    elif 'usedpermissionslist_' in feature_name_lower or 'requestedpermissionlist_' in feature_name_lower:
        return "permission"
    elif 'suspiciousapilist_' in feature_name_lower or 'restrictedapilist_' in feature_name_lower:
        return "api"
    elif 'intentfilterlist_' in feature_name_lower:
        return "filter"
    elif 'urldomainlist_' in feature_name_lower:
        return "url"
    else:
        return "unknown"

def extract_class_name_from_feature(feature_name):
    all=feature_name.split('.')
    class_name=all[len(all)-1]
    return class_name

def find_component_code(a, d, class_name):
    """查找组件对应的代码"""
    locations = []
    try:
        # 获取所有相关的类
        for class_analysis in d[0].get_classes():
            if class_name in class_analysis.get_name():
                # 获取类中的所有方法
                for method in class_analysis.get_methods():
                    locations.append({
                        'class': class_analysis.get_name(),
                        'method': method.get_name(),
                        "similar": 1
                    })

    except Exception as e:
        print(f"Error finding component code: {e}")
    return locations

def reverse_map_permission_to_apis(permission_feature,PMap):
    """将权限映射到相关的API集合"""
    permission_feature=permission_feature.lower()
    if 'permissionslist_' in permission_feature:
        permission=permission_feature.split('permissionslist_')[1]
    else:
        permission=permission_feature.split('usedpermissionslist_')[1]
    return PMap.getApis(permission.rsplit('.', 1)[0] + '.' + permission.rsplit('.', 1)[1].upper())

def find_permission_related_code(d, dx, api_set):
    """查找权限相关的代码"""
    locations = []
    for api in api_set:
        locations.extend(find_api_invocations(d, dx, [api]))
    return locations


def get_filter_translation_patterns(api_feature):
    return api_feature.replace('IntentFilterList_','')

def get_url_translation_patterns(api_feature):
    return api_feature.replace('URLDomainList_','')

def get_api_translation_patterns(api_feature):
    """获取API的包和名字"""
    return api_feature.split('_')[1]

def find_filter_operations(d, dx, patterns):
    matches = []
    for method in d[0].get_methods():
        g = dx.get_method(method)
        for BasicBlock in g.get_basic_blocks().get():
            Instructions = BasicBlockAttrBuilder.GetBasicBlockDalvikCode(BasicBlock)
            for instruction in Instructions:
                if "invoke-" in instruction:
                    pass
                else:
                    similarity = string_similarity(instruction, patterns)
                    matches.append({
                        "class": method.class_name,
                        "method": method.name,
                        "similar": similarity,
                    })
    max_similar_match = max(matches, key=lambda x: x["similar"])
    return max_similar_match

def find_url_operations(d, dx, patterns):
    matches = []
    for class_obj in d[0].get_classes():
        class_name=class_obj.get_name()
        class_name=class_name[1:len(class_name)-1]

        # 类级别的各种信息搜索
        class_info_to_check = [
            ("class_name", class_name),
            ("super_class", class_obj.get_superclassname() or ""),
        ]

        # 添加接口信息（如果存在）
        try:
            interfaces = class_obj.get_interfaces()
            if interfaces:
                class_info_to_check.append(("interfaces", ", ".join(interfaces)))
        except:
            pass

        # 添加源文件信息（如果存在）
        try:
            source_file = class_obj.get_source_file()
            if source_file:
                class_info_to_check.append(("source_file", source_file))
        except:
            pass


        for info_type, info_value in class_info_to_check:
            if info_value:
                similarity = string_similarity(info_value, patterns)
                if similarity > 0:
                    matches.append({
                        "class": class_name,
                        "method": '$',
                        "similar": similarity,
                    })
        for annotation in class_obj.get_annotations():
            annotation_similarity = string_similarity(annotation, patterns)
            if annotation_similarity > 0:
                matches.append({
                    "class": class_name,
                    "method": '$$',
                    "similar": annotation_similarity,
                })

        for field in class_obj.get_fields():
            field_info = {
                "name": field.get_name(),
                "type": field.get_descriptor(),
                "access_flags": field.get_access_flags_string(),
            }

            for key, value in field_info.items():
                if value:
                    similarity = string_similarity(value, patterns)
                    if similarity > 0:
                        matches.append({
                            "class": class_name,
                            "method": '$$$',
                            "similar": similarity,
                        })
        # 方法信息
        for method in class_obj.get_methods():
            method_name = method.get_name()

            # 方法基本信息
            method_info = {
                "name": method_name,
                "descriptor": method.get_descriptor(),
                "access_flags": method.get_access_flags_string(),
            }

            for key, value in method_info.items():
                if value:
                    similarity = string_similarity(value, patterns)
                    if similarity > 0:
                        matches.append({
                            "class": class_name,
                            "method": method_name,
                            "similar": similarity,
                        })

            # 方法内部指令
            try:
                g = dx.get_method(method)
                if g:
                    for BasicBlock in g.get_basic_blocks().get():
                        Instructions = BasicBlockAttrBuilder.GetBasicBlockDalvikCode(BasicBlock)
                        for instruction in Instructions:
                            instruction_str = str(instruction)
                            instruction_similarity = string_similarity(instruction_str, patterns)
                            if instruction_similarity > 0:
                                matches.append({
                                    "class": class_name,
                                    "method": method_name,
                                    "similar": instruction_similarity,
                                })
            except Exception as e:
                raise e
    max_similar_match = max(matches, key=lambda x: x["similar"])
    return max_similar_match
def find_api_invocations(d, dx, patterns):
    """查找API调用位置"""
    matches = []
    for method in d[0].get_methods():
        g = dx.get_method(method)
        for BasicBlock in g.get_basic_blocks().get():
            Instructions = BasicBlockAttrBuilder.GetBasicBlockDalvikCode(BasicBlock)
            for instruction in Instructions:
                if "invoke-" in instruction:
                    similarity = string_similarity(instruction, patterns)
                    matches.append({
                        "class": method.class_name,
                        "method": method.name,
                        "similar": similarity,
                    })
    max_similar_match = max(matches, key=lambda x: x["similar"])
    return max_similar_match
def find_http_operations(d, dx):
    """查找HTTP操作"""
    locations = []
    http_patterns = [
        'http://',
        'https://',
        'HttpURLConnection',
        'HttpClient',
        'WebView'
    ]

    try:
        for class_analysis in d.get_classes():
            for method in class_analysis.get_methods():
                method_code = method.get_code()
                if method_code:
                    instructions = method_code.get_bc().get_instructions()
                    for ins in instructions:
                        ins_str = str(ins)
                        for pattern in http_patterns:
                            if pattern in ins_str:
                                locations.append({
                                    'class_name': class_analysis.get_name(),
                                    'method_name': method.get_name()
                                })
    except Exception as e:
        print(f"Error finding HTTP operations: {e}")
    return locations

def decompile_smali_to_java(smali_code, method_name):
    """将smali代码转换为Java代码（简化实现）"""
    # 实际实现需要集成反编译器
    # 这里返回示例Java代码
    if method_name == "suspiciousMethod":
        return """
public void suspiciousMethod() {
    Log.d("MalwareDetection", "Suspicious operation detected");
    // 潜在恶意行为代码
}
"""
    return "// 无法反编译的代码"
from difflib import SequenceMatcher

def find_similar_apis(a, b, similarity_threshold=0.8):
    """
    在两个列表中查找相似的API
    :param a: 第一个API列表
    :param b: 第二个API列表
    :param similarity_threshold: 相似度阈值 (0-1)
    :return: 匹配的API对列表
    """
    matches = []

    for api_a in a:
        for api_b in b:
            # 计算字符串相似度
            similarity = SequenceMatcher(None, api_a, api_b).ratio()
            if similarity >= similarity_threshold:
                matches.append((api_a, api_b, similarity))

    # 按相似度排序
    matches.sort(key=lambda x: x[2], reverse=True)
    return matches


def splite_dict_2_level(feature_name, high_risk_locations, all_api, a, d, dx):
    try:
        # 添加缓存机制 - 避免重复查找相同的API
        api_cache = {}

        feature_type = determine_feature_type(feature_name)

        if feature_type == "component":
            class_name = extract_class_name_from_feature(feature_name)
            methods_all = find_component_code(a, d, class_name)
            if len(methods_all) > 0:
                high_risk_locations.append(methods_all)

        elif feature_type == "api":
            translated_patterns = get_api_translation_patterns(feature_name)

            # 使用缓存避免重复查找
            if translated_patterns in api_cache:
                max_similar_point = api_cache[translated_patterns]
            else:
                max_similar_point = find_api_invocations_optimized(d, dx, translated_patterns)
                api_cache[translated_patterns] = max_similar_point

            high_risk_locations.append(max_similar_point)

        elif feature_type == "filter":
            translated_patterns = get_filter_translation_patterns(feature_name)
            # 同样添加缓存
            if translated_patterns in api_cache:
                result = api_cache[translated_patterns]
            else:
                result = find_filter_operations(d, dx, translated_patterns)
                api_cache[translated_patterns] = result
            high_risk_locations.append(result)

        elif feature_type == "url":
            translated_patterns = get_url_translation_patterns(feature_name)
            # 同样添加缓存
            if translated_patterns in api_cache:
                result = api_cache[translated_patterns]
            else:
                result = find_filter_operations(d, dx, translated_patterns)
                api_cache[translated_patterns] = result
            high_risk_locations.append(result)
        #时间开销过大，暂去除
        # if feature_type == "permission":
        #     # 权限相关特征
        #     api_set = reverse_map_permission_to_apis(feature_name,PMap)
        #     high_risk_locations_tmp=[]
        #     similar_apis = find_similar_apis(all_api_clear, api_set, 0.5)
        #
        #     for _,one_api,_ in similar_apis:
        #         max_similar_point=find_api_invocations(d, dx, one_api)
        #         high_risk_locations_tmp.append(max_similar_point)
        #     tmp=heapq.nlargest(10, high_risk_locations_tmp, key=lambda x: x['similar'])
        #     high_risk_locations.extend(tmp)
        #     print(high_risk_locations)
    except Exception as e:
        print(f"Error in splite_dict_2_level: {e}")
    return high_risk_locations
def find_api_invocations_optimized(d, dx, patterns):
    """优化的API调用查找 - 减少不必要的计算"""
    best_match = {"class": "", "method": "", "similar": 0.0}

    for method in d[0].get_methods():
        g = dx.get_method(method)
        for BasicBlock in g.get_basic_blocks().get():
            Instructions = BasicBlockAttrBuilder.GetBasicBlockDalvikCode(BasicBlock)
            for instruction in Instructions:
                if "invoke-" in instruction:
                    # 先进行快速检查，避免不必要的相似度计算
                    if patterns in instruction:
                        similarity = 1.0  # 完全匹配
                    else:
                        # 只有当指令包含关键部分时才计算相似度
                        if any(keyword in instruction for keyword in patterns.split('/')[-2:]):
                            similarity = string_similarity(instruction, patterns)
                        else:
                            continue

                    if similarity > best_match["similar"]:
                        best_match = {
                            "class": method.class_name,
                            "method": method.name,
                            "similar": similarity
                        }

                    # 如果已经找到完美匹配，提前返回
                    if similarity >= 0.95:
                        return best_match

    return best_match

def get_file_path_by_seq(seq):
    try:
        conn=get_connection()
        with conn.cursor() as cursor:
            # 执行SQL查询
            sql = "SELECT name,path FROM app_label WHERE seq = %s"
            cursor.execute(sql, (seq,))
            # 获取结果
            result = cursor.fetchone()
            if result:
                return result[0],result[1]  # 返回路径
            else:
                return '',''
    finally:
        conn.close()


import subprocess
import tempfile
from javalang.tree import MethodDeclaration
import json

class SmaliDecompiler:
    def __init__(self, baksmali_path="smali-2.5.2.jar", jadx_path="jadx/bin/jadx"):
        self.baksmali_path = baksmali_path
        self.jadx_path = jadx_path

    def decompile_smali_to_java(self, smali_file_path, output_dir):
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. 将smali文件重新打包为dex文件
            dex_file = os.path.join(temp_dir, "classes.dex")
            self._assemble_smali_to_dex(smali_file_path, dex_file)
            # 2. 使用jadx将dex反编译为Java
            self._decompile_dex_to_java(dex_file, output_dir)
            # print(f"反编译完成，结果保存在: {output_dir}")
            return True

    def decompile_smali_directory(self, smali_dir_path, output_dir):
        """
        将整个smali目录反编译为Java代码
        """
        try:
            # 创建临时目录
            with tempfile.TemporaryDirectory() as temp_dir:
                # 1. 将整个smali目录打包为dex文件
                dex_file = os.path.join(temp_dir, "classes.dex")
                self._assemble_smali_dir_to_dex(smali_dir_path, dex_file)

                # 2. 使用jadx将dex反编译为Java
                self._decompile_dex_to_java(dex_file, output_dir)

                # print(f"反编译完成，结果保存在: {output_dir}")
                return True

        except Exception as e:
            print(f"反编译失败: {e}")
            return False

    def _assemble_smali_to_dex(self, smali_file_path, output_dex_path):
        """使用smali工具将smali文件汇编为dex"""
        # 需要先将单个smali文件放到临时目录中
        with tempfile.TemporaryDirectory() as temp_smali_dir:
            # 复制smali文件到临时目录
            shutil.copy(smali_file_path, temp_smali_dir)
            self._assemble_smali_dir_to_dex(temp_smali_dir, output_dex_path)

    def _assemble_smali_dir_to_dex(self, smali_dir_path, output_dex_path):
        """使用smali工具将smali目录汇编为dex"""
        cmd = [
            "java", "-jar", self.baksmali_path,
            "assemble", smali_dir_path,
            "-o", output_dex_path

        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or len(result.stderr)>0:
            raise Exception(f"smali汇编失败: {result.stderr}")

    def _decompile_dex_to_java(self, dex_file_path, output_dir):
        """使用jadx将dex文件反编译为Java"""
        cmd = [
            self.jadx_path,
            "-d", output_dir,
            "--no-imports",  # 不优化imports
            "--no-res",      # 不处理资源
            "--decompilation-mode", "restructure",  # 改为auto模式（或其他可用模式）
            "--show-bad-code",  # 显示有问题的代码
            dex_file_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or len(result.stderr)>0 or len(result.stderr)>0:
            if len(result.stderr)==0:
                raise Exception("jadx反编译失败:"+result.stdout)
            else:
                raise Exception("jadx反编译失败:"+result.stderr)

class JavaMethodExtractor:
    def extract_methods_from_java_content(self, java_content, class_name):
        """
        从Java代码内容中提取所有方法信息
        返回格式: {方法签名: 方法信息}
        """
        methods_dict = {}
        try:

            #替换 不支持的类声明
            pattern = r'new\s+(\d+)\('
            replacement = r'new a\1('
            java_content=re.sub(pattern, replacement, java_content)
            tree = javalang.parse.parse(java_content)

            for path, node in tree:
                if isinstance(node, MethodDeclaration):
                    method_signature = self._get_method_signature(node, class_name)
                    method_full_text = self._extract_full_method_text(java_content, node)

                    methods_dict[method_signature] = {
                        'name': node.name,
                        'return_type': str(node.return_type) if node.return_type else 'void',
                        'parameters': [self._format_parameter(param) for param in node.parameters],
                        'modifiers': list(node.modifiers) if hasattr(node, 'modifiers') else [],
                        'body': self._extract_method_body(java_content, node),
                        'full_method_text': method_full_text,  # 添加完整的函数文本
                        'class': class_name
                    }

        except Exception as e:
            print(f"解析Java内容失败: {e}")

        return methods_dict

    def _get_method_signature(self, method_node, class_name):
        """生成方法签名"""
        return_type = str(method_node.return_type) if method_node.return_type else 'void'
        params = ', '.join([str(param.type) for param in method_node.parameters])
        return f"{class_name}.{method_node.name}({params}):{return_type}"

    def _format_parameter(self, param):
        """格式化参数信息"""
        return {
            'name': param.name,
            'type': str(param.type),
            'modifiers': getattr(param, 'modifiers', [])
        }

    def _extract_method_body(self, java_content, method_node):
        """提取方法的完整函数体"""
        lines = java_content.split('\n')
        start_line = method_node.position.line - 1 if method_node.position else 0

        # 找到方法体的开始和结束
        brace_count = 0
        in_body = False
        body_lines = []

        for i in range(start_line, len(lines)):
            line = lines[i].strip()

            if not in_body and '{' in line:
                in_body = True
                brace_count += line.count('{')
                continue

            if in_body:
                brace_count += line.count('{')
                brace_count -= line.count('}')
                body_lines.append(lines[i])

                if brace_count <= 0:
                    break

        return '\n'.join(body_lines)

    def _extract_full_method_text(self, java_content, method_node):
        """
        提取完整的函数定义文本，包括修饰符、返回类型、方法名、参数和方法体
        """
        lines = java_content.split('\n')
        start_line = method_node.position.line - 1 if method_node.position else 0

        # 找到方法声明的开始行
        method_start_line = start_line
        while method_start_line > 0:
            prev_line = lines[method_start_line - 1].strip()
            # 如果上一行是空行或注释，继续向上查找
            if not prev_line or prev_line.startswith('//') or prev_line.startswith('/*') or prev_line.startswith('*'):
                method_start_line -= 1
            else:
                break

        # 找到方法体的结束
        brace_count = 0
        in_method = False
        method_lines = []

        for i in range(method_start_line, len(lines)):
            line = lines[i]

            # 检查是否进入方法体
            if not in_method and '{' in line:
                in_method = True
                brace_count += line.count('{')
            elif in_method:
                brace_count += line.count('{')
                brace_count -= line.count('}')

            method_lines.append(line)

            if in_method and brace_count <= 0:
                break

        return '\n'.join(method_lines)

class SetEncoder(json.JSONEncoder):
    """自定义JSON编码器，用于处理set类型"""
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)

def find_smali_file(smali_dir, class_name):
    """
    根据类名查找对应的smali文件
    """
    class_path = class_name[1:].replace('/', os.sep).replace(';', '') + '.smali'
    smali_file_path = os.path.join(smali_dir, class_path)

    if os.path.exists(smali_file_path):
        return smali_file_path
    else:
        for root, dirs, files in os.walk(smali_dir):
            for file in files:
                if file.endswith('.smali') and class_name[1:].replace('/', '$') in file:
                    return os.path.join(root, file)
        return None

def process_methods(apk_name, decompiler, extractor, methods, smali_dir, output_json_path, output_tmp):
    """
    处理所有methods，提取方法信息并保存到JSON
    """
    methods_ex_pack=[]
    for one in methods:
        if isinstance(one,dict):
            methods_ex_pack.append(one)
        if isinstance(one,list):
            methods_ex_pack.extend(one)
    methods=methods_ex_pack
    results = {}

    # 按类分组methods
    class_methods = {}
    for method in methods:
        class_name = method['class']
        method_name = method['method']

        if class_name not in class_methods:
            class_methods[class_name] = set()
        class_methods[class_name].add(method_name)

    # 处理每个类
    for class_name, target_method_names in class_methods.items():
        print(f"处理类: {class_name}")

        # 查找smali文件
        smali_file_path = find_smali_file(smali_dir, class_name)
        if not smali_file_path:
            print(f"未找到类 {class_name} 对应的smali文件")
            continue

        try:
            # 反编译smali文件为Java内容（不保存文件）
            java_content = decompiler.decompile_smali_to_java_content(apk_name, smali_file_path, output_tmp)

            if len(java_content) == 0:
                continue

            # 提取所有方法信息
            all_methods = extractor.extract_methods_from_java_content(java_content, class_name)

            # 筛选目标方法
            class_results = {}
            for method_signature, method_info in all_methods.items():
                if method_info['name'] in target_method_names:
                    class_results[method_signature] = method_info

            results[class_name] = {
                'smali_file': smali_file_path,
                'target_method_count': len(target_method_names),
                'found_method_count': len(class_results),
                'methods': class_results
            }

            print(f"类 {class_name} 找到 {len(class_results)}/{len(target_method_names)} 个目标方法")

        except Exception as e:
            print(f"处理类 {class_name} 时出错: {e}")
            continue

    # 保存结果到JSON文件
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, cls=SetEncoder)
        print(f"结果已保存到: {output_json_path}")
    except Exception as e:
        print(f"保存JSON文件失败: {e}")

    return results

from openai import OpenAI
def deepseek_api(promt_str):
    client = OpenAI(api_key="sk-837765b3c48740df810a04b8f27adabe", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            # {"role": "system", "content": "You are a professional Android security analyst"},
            {"role": "user", "content": promt_str},
        ],
        stream=False
    )
    return response.choices[0].message.content
def deepseek_api_chat(content, task):
    client = OpenAI(api_key="sk-837765b3c48740df810a04b8f27adabe", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": content},
            {"role": "user", "content": task}
        ],
        stream=False
    )
    return response.choices[0].message.content
def extract_full_method_texts(json_data):
    full_method_texts = []
    for class_data in json_data.values():
        methods = class_data.get('methods', {})
        for method_info in methods.values():
            full_method_text = method_info.get('full_method_text')
            if full_method_text:
                full_method_texts.append(full_method_text)
    return '\n'.join(full_method_texts)



import requests
import time

def llm_chat(seq,old_talk, task, model,t=None,l=None):
    if model=='deepseek-chat' or model=='deepseek-reasoner':
        talk_list, content=deepseek_api_chat_talks(old_talk, task)
    else:
        talk_list, content=ollama_api_chat_talks(old_talk, task, model,t,l)
    return talk_list, content

def get_label_loop(llm_ret, model_name):
    label_scores = []
    for i in range(15):
        prompt = 'Choose one word from [safe,unknown,normal, benign, risky, vulnerable, malicious, malware] to summarize the following sentence.\r\n' + llm_ret
        talks = []
        _, tmp = llm_chat('', talks, prompt, model_name)
        label = convert_label_with_weight(tmp)  # 修改为带权重的转换函数
        label_scores.append(label)

    # 计算平均分数
    avg_score = sum(label_scores) / len(label_scores) if label_scores else 0
    return 1 if avg_score >= 0.5 else 0  # 阈值保持0.5

def convert_label_with_weight(label_str):
    """带权重的标签转换函数"""
    label_str = label_str.strip().lower()

    # 定义标签权重（0=安全，1=恶意）
    label_weights = {
        'safe': 0.0,
        'normal': 0.0,
        'benign': 0.0,
        'unknown': 0.2,
        'vulnerable': 0.2,
        'risky': 0.3,
        'malware': 1.0,
        'malicious': 1.0
    }

    # 检查标签是否在权重字典中
    for label, weight in label_weights.items():
        if label in label_str:
            return weight
    # 默认返回中间值
    return 0.5
def ollama_api(prompt, model="gemma3:1b"):
    url = "http://211.65.82.10:8087/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=data, timeout=90)
        # print(f"HTTP状态码: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            return result["response"]
        else:
            return f"错误: {response.status_code} - {response.text}"
    except Exception as e:
        return f"请求异常: {e}"
def ollama_api_chat(background, task, model="gemma3:1b"):
    url = "http://211.65.82.10:8087/api/chat"

    # 第一次通信：发送背景信息
    background_data = {
        "model": model,
        "messages": [
            {"role": "user", "content": background}
        ],
        "stream": False
    }

    try:
        # 发送背景信息
        response = requests.post(url, json=background_data, timeout=60)

        if response.status_code == 200:
            # 获取背景信息的响应
            background_response = response.json()

            # 第二次通信：在同一会话中发送任务
            task_data = {
                "model": model,
                "messages": [
                    {"role": "user", "content": background},
                    {"role": "assistant", "content": background_response.get("message", {}).get("content", "")},
                    {"role": "user", "content": task}
                ],
                "stream": False
            }

            # 发送任务
            task_response = requests.post(url, json=task_data, timeout=60)

            if task_response.status_code == 200:
                result = task_response.json()
                return result.get("message", {}).get("content", "")
            else:
                return f"任务请求错误: {task_response.status_code} - {task_response.text}"
        else:
            return f"背景信息请求错误: {response.status_code} - {response.text}"

    except Exception as e:
        return f"请求异常: {e}"




def get_xml(seq,dir):
    apkName,apkPath=get_file_path_by_seq(seq)
    # 遍历所有.py文件
    xml_file=dir + os.sep + apkName.replace('.', '_')+ os.sep +'AndroidManifest.xml'
    content=None
    if os.path.exists(xml_file):
        content=open(xml_file, encoding='utf-8').read()
    return content

def deepseek_api_chat_talks(old_talk, task, model="deepseek-chat"):
    client = OpenAI(api_key="sk-837765b3c48740df810a04b8f27adabe", base_url="https://api.deepseek.com")

    talk_list = old_talk
    content = ''

    if len(task) > 0:
        talk_list.append({"role": "user", "content": task})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=talk_list,
            stream=False
        )

        content = response.choices[0].message.content
        talk_list.append({"role": "assistant", "content": content})

    except Exception as e:
        print(f"DeepSeek API请求异常: {e}")

    return talk_list, content





def analyze_risk_components(seq, risk_nodes,jadx_path,smali_path):
    risk_nodes_smali = revert_smali_batch(risk_nodes)

    # 转换为Java代码
    RM_java = smali_2_java(jadx_path, seq, smali_path, risk_nodes_smali)

    # 转换为字符串
    RM_str = list_2_str(RM_java)
    return RM_str, '', '', ''
def build_feature_based_prompt(all_feature):
    """构建基于关键特征的提示词"""
    if not all_feature:
        return "No significant features detected."

    prompt_parts = []

    # 分类特征类型
    permission_features = []
    api_features = []
    component_features = []
    url_features = []
    filter_features = []

    for feature in all_feature:
        feature_lower = feature.lower()
        if 'permission' in feature_lower:
            permission_features.append(feature)
        elif 'api' in feature_lower:
            api_features.append(feature)
        elif any(comp in feature_lower for comp in ['activity', 'service', 'receiver', 'provider']):
            component_features.append(feature)
        elif 'url' in feature_lower:
            url_features.append(feature)
        elif 'filter' in feature_lower:
            filter_features.append(feature)
        else:
            # 默认分类
            permission_features.append(feature)

    # 构建分类特征描述
    if permission_features:
        prompt_parts.append("## Permission Features:")
        for feature in permission_features[:10]:  # 限制数量
            prompt_parts.append(f"- {feature}")

    if api_features:
        prompt_parts.append("## API Features:")
        for feature in api_features[:10]:
            prompt_parts.append(f"- {feature}")

    if component_features:
        prompt_parts.append("## Component Features:")
        for feature in component_features[:10]:
            prompt_parts.append(f"- {feature}")

    if url_features:
        prompt_parts.append("## URL Features:")
        for feature in url_features[:5]:
            prompt_parts.append(f"- {feature}")

    if filter_features:
        prompt_parts.append("## Intent Filter Features:")
        for feature in filter_features[:5]:
            prompt_parts.append(f"- {feature}")

    return "\n".join(prompt_parts)

def get_llm_label_chat_talks(seq, AM_all, AI_all, RM_all, CM_all, model_name, all_feature, taskKernal_last, mali_p):
    # 第一阶段：基于模型标签mali_p的推理
    phase1_prompt = build_phase1_prompt(mali_p)
    task = 'Role: You are an Android malware analyst. Task: Analyze the initial malware probability and provide preliminary judgment.\n\n'
    content_1 = 'Initial Malware Probability Analysis:\n' + phase1_prompt

    # 第一阶段对话
    talks = [
        {"role": "user", "content": content_1},
    ]
    talks, phase1_ret = llm_chat(seq, talks, task, model_name)
    final_label = get_label_loop(phase1_ret,model_name)
    # 第二阶段：基于可疑特征和权重分数的详细推理
    feature_prompt = build_phase2_prompt(all_feature)
    task = 'Task: Conduct detailed analysis combining initial judgment with suspicious features.\n\n ' \
           'Detailed Feature Analysis:\n' + feature_prompt

    # 第二阶段对话
    talks, phase2_ret = llm_chat(seq, talks, task, model_name)
    final_label = get_label_loop(phase2_ret,model_name)
    # 第三阶段：基于代码结构的深度分析（新增）
    if len(RM_all)>0:
        talks.append( {"role": "user", "content": background('', '', RM_all, '')})
        code_analysis_prompt = build_phase3_code_prompt(AM_all, AI_all, RM_all, CM_all)
        task = 'Task: Analyze the code structure and behavioral patterns to validate previous findings.\n\n' \
               'Code Structure Analysis:\n' + code_analysis_prompt
        talks, phase3_ret = llm_chat(seq, talks, task, model_name)
        final_label = get_label_loop(phase3_ret, model_name)  # 基于第三阶段结果
        if final_label==0:
            if len(AM_all)+len(AI_all)>0:
                talks.append({"role": "user", "content": background(AM_all, AI_all, '', '')})
                talks,phase3_ret=llm_chat(seq,talks, task,model_name)
                final_label = get_label_loop(phase3_ret,model_name)

    # 保存完整的对话记录（包含三个阶段）
    prompt_dir = r'.' + os.sep + 'decompiled_java' + os.sep + 'prompt_dir_4'
    with open(prompt_dir + os.sep + str(seq) + ".txt", 'w') as f:
        f.write(str(talks))

    return final_label, phase3_ret


def background(AM_all, AI_all, RM_all, CM_all):
    return f"""
    ### Key Code Segments:
{get_code_snippets(AM_all, AI_all, RM_all, CM_all)}
    """


def build_phase3_code_prompt(AM_all, AI_all, RM_all, CM_all):
    """构建第三阶段代码分析提示词（最小化改动）"""

    # 简化的代码分析
    am_summary = f"Trigger Components: {len(AM_all.splitlines()) if AM_all else 0} lines"
    ai_summary = f"Trigger Paths: {len(AI_all.splitlines()) if AI_all else 0} lines"
    rm_summary = f"Risk Methods: {len(RM_all.splitlines()) if RM_all else 0} lines"
    cm_summary = f"Called Methods: {len(CM_all.splitlines()) if CM_all else 0} lines"

    return f"""
## Code Structure Analysis:

### Component Flow:
- {am_summary}
- {ai_summary}
- {rm_summary}
- {cm_summary}

## Analysis Requirements:
1. Analyze the code execution flow from components to risk methods
2. Identify any suspicious API call patterns
3. Evaluate the relationship between trigger paths and risk methods
4. Assess if the code structure indicates malicious behavior

## Output Format:
[Code Analysis]
Execution Flow: [Describe the component->API->risk method flow]
Suspicious Patterns: [List any malicious code patterns found]
Technical Evidence: [Specific code-level indicators]
Final Assessment: [Benign/Malicious based on code analysis]
"""

def get_code_snippets(AM_all, AI_all, RM_all, CM_all):
    """提取关键代码片段"""
    snippets = []

    if AM_all and len(AM_all.strip()) > 0:
        am_lines = AM_all.split('\n')
        snippets.append("Trigger Components:\n" + '\n'.join(am_lines))

    if AI_all and len(AI_all.strip()) > 0:
        ai_lines = AI_all.split('\n')
        snippets.append("Trigger APIs:\n" + '\n'.join(ai_lines))

    if RM_all and len(RM_all.strip()) > 0:
        rm_lines = RM_all.split('\n')
        snippets.append("Risk Methods:\n" + '\n'.join(rm_lines))

    if CM_all and len(CM_all.strip()) > 0:
        cm_lines = CM_all.split('\n')
        snippets.append("Called Methods:\n" + '\n'.join(cm_lines))

    return '\n\n'.join(snippets) if snippets else "No significant code segments available."
def build_phase1_prompt(mali_p):
    """构建第一阶段基于模型标签的提示词"""
    return f"""
## Initial Machine Learning Assessment:
{'Malicious' if mali_p>0.5 else 'Benign'}

## Phase 1 Analysis Requirements:
1. Evaluate the malware probability score
2. Provide preliminary maliciousness judgment based on the probability
3. Assess the confidence level of this initial assessment

## Output Format:
[Phase 1 Analysis]
Preliminary Judgment: [Benign/Suspicious/Malicious]
Probability Assessment: [Analysis of the {mali_p:.4f} score]
Initial Confidence: [High/Medium/Low]
"""

def build_phase2_prompt(all_feature):
    """构建第二阶段基于特征和权重的提示词"""
    feature_prompt = build_feature_based_prompt(all_feature)

    return f"""
## Suspicious Features and Weight Analysis:
{feature_prompt}

## Phase 2 Analysis Requirements:
1. Combine Phase 1 judgment with detailed feature analysis
2. Analyze the most suspicious features and their weights
3. Validate or revise the initial judgment based on feature evidence
4. Provide final classification with detailed reasoning

## Output Format:
[Phase 2 Analysis]
Final Classification: [Benign/Malicious]
Key Suspicious Features: [List 2-3 most important suspicious features]
Feature Weight Analysis: [Analysis of feature importance and weights]
Final Reasoning: [Complete analysis combining probability and features]
Confidence Level: [High/Medium/Low]
"""


import requests
import threading
import time

endpoints = [

    "http://211.65.82.10:8085",  # GPU0
    "http://211.65.82.10:8086",  # GPU1
    "http://211.65.82.10:8087",   # GPU2
    "http://211.65.82.10:8088"  # GPU3
]

endpoint_locks = {ep: threading.Lock() for ep in endpoints}

endpoint_round_robin = itertools.cycle(endpoints)
round_robin_lock = threading.Lock()

def ollama_api_chat_talks(old_talk, task, model="codellama:7b", t=0.8, l=None):
    talk_list = old_talk.copy() if old_talk else []
    if len(task) > 0:
        talk_list.append({"role": "user", "content": task})

    task_data = {
        "model": model,
        "messages": talk_list,
        "stream": False,
        "keep_alive": 15,
        "options": {"num_gpu": 99999}
    }
    if t is not None:
        task_data['temperature'] = t
    if l is not None:
        task_data['options']['num_ctx'] = l*1000

    # 轮询获取下一个endpoint
    with round_robin_lock:
        ep = next(endpoint_round_robin)

    url = f"{ep}/api/chat"
    with endpoint_locks[ep]:
        while True:
            try:
                task_response = requests.post(url, json=task_data, timeout=90)
                if task_response.status_code == 200:
                    result = task_response.json()
                    content = result.get("message", {}).get("content", "")
                    talk_list.append({"role": "assistant", "content": content})
                    return talk_list, content
                else:
                    print(f"服务 {ep} 返回错误码: {task_response.status_code}")
                    # 错误时切换到下一个服务
                    with round_robin_lock:
                        ep = next(endpoint_round_robin)
                    url = f"{ep}/api/chat"
            except Exception as e:
                print(f"服务 {ep} 请求异常: {e}")
                with round_robin_lock:
                    ep = next(endpoint_round_robin)
                url = f"{ep}/api/chat"

            print("切换服务，1秒后重试...")
            time.sleep(1)
def extract_java_code(text):
    pattern = r'```(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    content=''
    for one in matches:
        content=one
    return content


def get_ref_code(RM,model):
    task= """
    Role: You are a software refactoring engineer. 
    TASK: Refactor the following code and print only one refactored code. 
    Using these operations randomly: Extract Method, Inline, Move, Change Statement Order
        """
    talks=[
        {"role": "user", "content": RM},
    ]
    talks,llm_ret=llm_chat(talks, task,model)
    java_code=extract_java_code(llm_ret)
    if len(java_code.strip())<=0:
        java_code=llm_ret
    return java_code

def convert_label(llm_ret):
    label = 0
    llm_ret=llm_ret.lower()
    llm_ret=llm_ret.split('.')[0]
    if 'mal' in llm_ret:
        label = 1
    if 'not' in llm_ret:
        label = 0
    if 'begin' in llm_ret:
        label = 0
    if 'safe' in llm_ret:
        label = 0
    # if 'risky' in llm_ret:
    #     label = 1
    if 'normal'  in llm_ret:
        label = 0
    return label


def is_list_empty(lst):
    """
    递归判断一个列表是否为空。
    空的条件：列表本身为空，或者所有元素（如果是列表）也都是空的。
    """
    # 如果根本不是列表，直接返回 False
    if not isinstance(lst, list):
        return False

    # 如果是最外层的空列表，返回 True
    if lst == []:
        return True

    # 遍历列表中的每个元素
    for item in lst:
        # 如果元素本身不是一个空列表
        if not isinstance(item, list):
            return False # 包含非列表元素，肯定不空
        # 如果元素是一个列表，递归检查它是否为空
        if not is_list_empty(item):
            return False # 如果子列表非空，则整个列表非空

    # 所有子列表都是空的，所以这个列表也是“空”的
    return True


from typing import List, Dict, Optional

class CSVDataLoader:
    def __init__(self, file_path: str):
        """
        初始化数据加载器
        :param file_path: CSV文件路径
        """
        self.file_path = file_path
        self.data_list = []  # 存储所有数据的列表
        self.seq_dict = {}   # 用于快速检索的字典，key为seq，value为对应行数据

    def load_data(self) -> List[Dict]:
        """
        加载CSV文件数据到列表和字典
        :return: 数据列表
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(self.file_path)

            # 将DataFrame转换为字典列表
            self.data_list = df.to_dict('records')

            # 构建seq到数据的映射字典，用于快速检索
            self.seq_dict = {item['seq']: item for item in self.data_list if pd.notna(item.get('seq'))}

            print(f"成功加载 {len(self.data_list)} 条数据")
            return self.data_list

        except Exception as e:
            print(f"加载文件失败: {e}")
            return []

    def get_label_guess_by_seq(self, seq: int) -> Optional[int]:
        if seq in self.seq_dict:
            item = self.seq_dict[seq]
            print(f"发现到seq为 {seq} 的缓存数据")
            return item.get('label_guess'),item.get('method')
        else:
            print(f"未找到seq为 {seq} 的数据")
            return None,None

    def get_full_data_by_seq(self, seq: int) -> Optional[Dict]:
        """
        根据seq检索并返回完整数据
        :param seq: 要检索的序列号
        :return: 完整数据字典，如果未找到返回None
        """
        if seq in self.seq_dict:
            return self.seq_dict[seq]
        else:
            print(f"未找到seq为 {seq} 的数据")
            return None

    def get_all_data(self) -> List[Dict]:
        """
        获取所有数据
        :return: 所有数据的列表
        """
        return self.data_list


def search_matching_nodes(graph, feature_list,feature_weight):
    matching_nodes = {}
    # 特征前缀与节点属性的映射关系

    fea=['usedpermissionslist_','requestedpermissionlist_', 'restrictedapilist_', 'suspiciousapilist_', 'activitylist_', 'servicelist_', 'broadcastreceiverlist_', 'intentfilterlist_', 'contentproviderlist_', 'hardwarecomponentslist_', 'urldomainlist_']

    attr=['permission', 'permission', 'sensitiveApi', 'suspiciousApi', 'name', 'name', 'name', 'filterToken', 'provider', 'hardware_component', 'url']
    search={}

    #遍历特征
    #判断特征类别
    for feature,weight in zip(feature_list,feature_weight):
        # 将特征值转换为小写用于不区分大小写比较
        original_feature = feature
        feature_lower = feature.lower()

        #若为权限
        if feature_lower.startswith(fea[0].lower()) or feature_lower.startswith(fea[1].lower()):
            feature_value = original_feature.replace(fea[0], '').replace(fea[1], '')
            search[attr[0]] = [feature_value.lower(),weight]  # 存储小写版本用于比较
        #若为服务
        elif feature_lower.startswith(fea[5].lower()):
            feature_value = original_feature.replace(fea[5], '')
            feature_value = 'L' + feature_value.replace('.', '/') + ';->'
            search[attr[5]] = [feature_value.lower(),weight]
        #若为广播
        elif feature_lower.startswith(fea[6].lower()):
            feature_value = original_feature.replace(fea[6], '')
            feature_value = 'L' + feature_value.replace('.', '/') + ';->'
            search[attr[6]] = [feature_value.lower(),weight]
        #若为活动
        elif feature_lower.startswith(fea[4].lower()):
            feature_value = original_feature.replace(fea[4], '')
            feature_value = 'L' + feature_value.replace('.', '/') + ';->'
            search[attr[4]] = [feature_value.lower(),weight]
        #若为敏感api
        elif feature_lower.startswith(fea[2].lower()):
            feature_value = original_feature.replace(fea[2], '')
            search[attr[2]] = [feature_value.lower(),weight]
        #若为可疑api
        elif feature_lower.startswith(fea[3].lower()):
            feature_value = original_feature.replace(fea[3], '')
            search[attr[3]] = [feature_value.lower(),weight]
        #若为url
        elif feature_lower.startswith(fea[10].lower()):
            feature_value = original_feature.replace(fea[10], '')
            search[attr[10]] = [feature_value.lower(),weight]
        #若为过滤器
        elif feature_lower.startswith(fea[7].lower()):
            feature_value = original_feature.replace(fea[7], '')
            search[attr[7]] = [feature_value.lower(),weight]
        #若为内容
        elif feature_lower.startswith(fea[8].lower()):
            feature_value = original_feature.replace(fea[8], '')
            search[attr[8]] = [feature_value.lower(),weight]
        #若为硬件组件
        elif feature_lower.startswith(fea[9].lower()):
            feature_value = original_feature.replace(fea[9], '')
            search[attr[9]] = [feature_value.lower(),weight]

    for node in graph.nodes():
        match_count = 0
        # 检查每个类别的匹配
        for attr_name, feature_value in search.items():
            node_attributes = getattr(node, attr_name, [])
            if len(node_attributes) > 0:
                if isinstance(node_attributes, list):
                    # 对于列表中的每个元素，转换为小写进行比较
                    for attr_value in node_attributes:
                        if isinstance(attr_value, str) and feature_value[0] in attr_value.lower():
                            match_count += feature_value[1]
                            break  # 找到一个匹配就计数一次
                # 对于字符串属性，检查是否包含特征值
                elif isinstance(node_attributes, str):
                    if feature_value[0] in node_attributes.lower():
                        match_count += feature_value[1]
        if match_count > 0:
            matching_nodes[node] = match_count
    return matching_nodes

def get_smali(smali_tmp,baksmali_path,apkPath,apkName):
    command = r'cd ' + smali_tmp + ' & java -jar ' + baksmali_path + ' d ' + apkPath
    if system == "Linux":
        command = 'cd ' + smali_tmp + '  && java -jar ' + baksmali_path + ' d ' + apkPath
    smali_dir = smali_tmp + os.sep + apkName.replace('.', '_')
    smali_status = True
    if not os.path.isdir(smali_dir):
        print(command)
        a = os.system(command)  # 使用a接收返回值
        if int(a) == 0:
            os.rename(smali_tmp + os.sep + 'out', smali_dir)  # 子文件夹重命名
        else:
            print('未进行解包，解包结果', a)
            smali_status = False
    return smali_dir,smali_status
import javalang

def analyze_java_code(java_code):
    """使用正则表达式简单检测非空方法和统计行数"""
    total_lines = java_code.count('\n') + 1

    # 检测包含语句的方法（方法体内有分号）
    has_non_empty_methods = True if java_code.count(';') else False

    return has_non_empty_methods, total_lines

def clear(AM_all, AI_all, RM_all, CM_all):
    classes=[AM_all,AI_all,RM_all,CM_all]
    classes_clear=[]
    line_all=0
    has_methods_all=False
    for one in classes:
        one = '\n'.join(line for line in one.split('\n') if '/* loaded from:' not in line)
        has_methods, lines = analyze_java_code(one)
        line_all+=lines
        if has_methods:
            has_methods_all=True
        classes_clear.append(one)
    if has_methods_all :
        return classes_clear[0],classes_clear[1],classes_clear[2],classes_clear[3],
    else:
        return '','','',''
import concurrent.futures





def process_single_item(idx, seq, explainer, feature_names, features_test, label_guess, feature_num, llm_name, recheck_num, device,mali_p):
    key_features = {}
    new_label,code_lines = get_label_llm(explainer, seq, feature_names, key_features, features_test, idx, label_guess, feature_num, llm_name, recheck_num, device,mali_p)
    return new_label,code_lines
import platform
import re
from typing import Dict, List


def parse_dot_file(file_path: str) -> Dict[str, List[str]]:
    """
    解析DOT文件，提取调用关系

    Args:
        file_path: DOT文件路径

    Returns:
        字典，key为调用者，value为被调用者列表
    """
    call_graph = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 提取所有的边（调用关系）
    edges = re.findall(r'\"(.+?)\"->\"(.+?)\"', content)

    for caller, callee in edges:
        if caller not in call_graph:
            call_graph[caller] = []
        if callee not in call_graph[caller]:  # 避免重复添加
            call_graph[caller].append(callee)

    return call_graph


def convert_special_method(method: str) -> str:
    """
    处理特殊方法（构造函数和静态初始化块）
    """
    if ' <init>' in method:
        # 构造函数: <com.example.Class: void <init>()>
        parts = method.split(' ')
        class_name = parts[0].strip('<>')
        smali_class = convert_class_to_smali(class_name)
        return f"{smali_class}-><init>()V"
    elif ' <clinit>' in method:
        # 静态初始化块: <com.example.Class: void <clinit>()>
        parts = method.split(' ')
        class_name = parts[0].strip('<>')
        smali_class = convert_class_to_smali(class_name)
        return f"{smali_class}-><clinit>()V"
    else:
        return method


def convert_type_to_smali(java_type: str) -> str:
    """
    将Java类型转换为Smali类型

    Args:
        java_type: Java类型，如 "void", "int", "java.lang.String[]"

    Returns:
        Smali类型
    """
    # 基本类型映射
    primitive_map = {
        'void': 'V',
        'boolean': 'Z',
        'byte': 'B',
        'char': 'C',
        'short': 'S',
        'int': 'I',
        'long': 'J',
        'float': 'F',
        'double': 'D'
    }

    # 去除空格
    java_type = java_type.strip()

    # 处理数组类型
    array_depth = 0
    base_type = java_type
    while base_type.endswith('[]'):
        array_depth += 1
        base_type = base_type[:-2]

    # 基本类型
    if base_type in primitive_map:
        smali_type = primitive_map[base_type]
    # 引用类型
    else:
        # 处理泛型（简单去除泛型参数）
        if '<' in base_type:
            base_type = base_type.split('<')[0]
        smali_type = f'L{base_type.replace(".", "/")};'

    # 添加数组维度
    smali_type = '[' * array_depth + smali_type

    return smali_type
def run_data_flow_analysis(apk_path, android_sdk, output_dir, soot_jar, sources_sinks_file=None):
    """
    执行数据流分析（污点分析）
    """
    print(f"[*] 开始数据流分析: {apk_path}")
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 构建FlowDroid命令
    cmd = [
        'java', '-jar', soot_jar,
        '-a', apk_path,
        '-p', android_sdk,
        '-s', sources_sinks_file,
        '-o', os.path.join(output_dir, 'taint_results.xml'),
        '--aliasflowins',
        '--nocallbacks',
        '--noexceptions',
        '--pathalgo', 'CONTEXTINSENSITIVE',
        '--pathreconstructionmode', 'fast',
        '--staticmode', 'CONTEXTFLOWINSENSITIVE',
    ]

    # 如果是Windows系统，添加额外的参数
    if platform.system() == "Windows":
        cmd.extend(['-dt', 'windows'])
    print(f"[*] 执行命令: {' '.join(cmd)}")
    # 执行命令
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600,
        encoding='utf-8',
        errors='ignore'
    )

    # 检查结果文件
    results_file = os.path.join(output_dir, 'taint_results.xml')
    if os.path.exists(results_file):
        return results_file
    else:

        return ''

def parse_taint_results(results_file, output_dir):
    """
    解析污点分析结果（支持XML格式）
    """
    print(f"[*] 解析污点分析结果: {results_file}")

    try:

        # 读取文件内容
        with open(results_file, 'r', encoding='utf-8') as f:
            content = f.read()
        # 检查文件内容
        if not content.strip():
            print(f"[-] 结果文件为空")
            return
        results = parse_xml_results(content)
        return results

    except Exception as e:
        print(f"[-] 解析结果失败: {e}")
        import traceback
        traceback.print_exc()
        return None
def parse_xml_results(xml_content):
    """
    解析XML格式的污点分析结果
    """
    try:
        import xml.etree.ElementTree as ET

        # 解析XML
        root = ET.fromstring(xml_content)

        # 初始化结果结构
        results_data = {
            "results": [],
            "performance": {},
            "summary": {
                "total_results": 0,
                "sources_count": 0,
                "sinks_count": 0
            }
        }

        # 解析性能数据
        perf_data = root.find('.//PerformanceData')
        if perf_data is not None:
            for entry in perf_data.findall('.//PerformanceEntry'):
                name = entry.get('Name', '')
                value = entry.get('Value', '')
                results_data["performance"][name] = value

        # 解析每个结果
        results_element = root.find('.//Results')
        if results_element is not None:
            for result_elem in results_element.findall('.//Result'):
                result = {
                    "sink": {},
                    "sources": []
                }

                # 解析sink
                sink_elem = result_elem.find('.//Sink')
                if sink_elem is not None:
                    result["sink"] = {
                        "statement": sink_elem.get('Statement', ''),
                        "method": sink_elem.get('Method', '')
                    }

                    # 解析sink的access path
                    access_path = sink_elem.find('.//AccessPath')
                    if access_path is not None:
                        result["sink"]["access_path"] = {
                            "value": access_path.get('Value', ''),
                            "type": access_path.get('Type', '')
                        }

                # 解析sources
                sources_elem = result_elem.find('.//Sources')
                if sources_elem is not None:
                    for source_elem in sources_elem.findall('.//Source'):
                        source = {
                            "statement": source_elem.get('Statement', ''),
                            "method": source_elem.get('Method', '')
                        }

                        # 解析source的access path
                        access_path = source_elem.find('.//AccessPath')
                        if access_path is not None:
                            source["access_path"] = {
                                "value": access_path.get('Value', ''),
                                "type": access_path.get('Type', '')
                            }

                        # 解析污点传播路径
                        taint_path = source_elem.find('.//TaintPath')
                        if taint_path is not None:
                            path_elements = []
                            for path_elem in taint_path.findall('.//PathElement'):
                                path_elements.append({
                                    "statement": path_elem.get('Statement', ''),
                                    "method": path_elem.get('Method', '')
                                })
                            source["taint_path"] = path_elements

                        result["sources"].append(source)

                results_data["results"].append(result)

        # 更新摘要信息
        results_data["summary"]["total_results"] = len(results_data["results"])
        results_data["summary"]["sources_count"] = sum(len(r["sources"]) for r in results_data["results"])
        results_data["summary"]["sinks_count"] = len(results_data["results"])

        return results_data

    except Exception as e:
        print(f"[-] 解析XML失败: {e}")
        return {"results": [], "performance": {}, "summary": {}}
def parse_flow_results(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except:
        # 如果文件是字符串形式
        root = ET.fromstring(xml_file)

    results = []

    # 遍历每个Result节点
    for result in root.findall('.//Result'):
        sink_info = result.find('Sink')
        sources = result.findall('Sources/Source')

        if sink_info is not None and sources:
            sink_method = sink_info.get('Method', '').strip('<>')

            for source in sources:
                source_method = source.get('Method', '').strip('<>')

                # 添加到结果列表
                results.append({
                    'caller': sink_method,      # 调用方（sink位置）
                    'callee': source_method     # 被调用方（source位置）
                })

    return results

def analyze_date_dependency(android_sdk,apkPath,taint_output_dir,soot_jar,SourcesAndSinks):
    taint_file = run_data_flow_analysis(
        apk_path=apkPath,
        android_sdk=android_sdk,
        output_dir=taint_output_dir,
        soot_jar=soot_jar,
        sources_sinks_file=SourcesAndSinks
    )
    flows = parse_flow_results(taint_file)
    graph={}
    for i,flow in enumerate(flows, 1):
        smali_caller = convert_to_smali_format(flow['caller'])
        # 对被调用方法排序
        tmp=[]
        smali_callee = convert_to_smali_format(flow['callee'])
        tmp.append(smali_callee.replace(':',''))
        graph[smali_caller.replace(':','')]=tmp
    return graph
def analyze_call_graph(android_sdk,soot_jar,seq,apkPath):
    seq=str(seq)
    """
    分析调用图并输出Smali格式的结果
    """
    dot_output_dir = r"."+os.sep+"decompiled_java"+os.sep+"dot_output"
    os.makedirs(dot_output_dir, exist_ok=True)

    # 生成唯一的 dot 文件名
    apk_basename = os.path.basename(apkPath).replace('.apk', '')
    dot_file_path = os.path.join(dot_output_dir, seq)

    if not os.path.exists(dot_file_path+".dot"):
        # 执行 Jadx 命令
        jadx_command = f'java -cp .;'+soot_jar+' App '+apkPath+' '+android_sdk+' ./ '+dot_file_path
        system = platform.system()
        if system == "Linux":
            jadx_command = f'java -cp .:'+soot_jar+' App '+apkPath+' '+android_sdk+' ./ '+dot_file_path
        print(f"执行 Jadx 命令生成调用图...")
        print(jadx_command)
        result = os.system(jadx_command)

        if result != 0:
            print(f"Jadx 命令执行失败，返回码: {result}")
            return {}


    # 使用第一个找到的 dot 文件
    actual_dot_file = dot_file_path+".dot"
    call_graph = parse_dot_file(actual_dot_file)

    # print("\n" + "="*60)
    # print("调用关系分析结果 (Smali格式)")
    # print("="*60 + "\n")

    # 统计信息
    total_calls = sum(len(callees) for callees in call_graph.values())
    unique_methods = set()
    for caller, callees in call_graph.items():
        unique_methods.add(caller)
        unique_methods.update(callees)

    # print(f"总调用关系数量: {total_calls}")
    # print(f"调用者数量: {len(call_graph)}")
    # print(f"唯一方法数量: {len(unique_methods)}")
    # print()

    # 按调用者排序输出
    sorted_callers = sorted(call_graph.keys())
    graph={}
    for caller in sorted_callers:
        callees = call_graph[caller]
        smali_caller = convert_to_smali_format(caller)
        # 对被调用方法排序
        sorted_callees = sorted(callees)
        tmp=[]
        for callee in sorted_callees:
            smali_callee = convert_to_smali_format(callee)
            tmp.append(smali_callee.replace(':',''))
        graph[smali_caller.replace(':','')]=tmp
    return graph

def add_edges_from_dict(graph_a, dict_b):
    """
    将字典b中的边添加到图a中，不新增节点，处理<init>节点

    参数:
    graph_a: networkx图对象
    dict_b: 格式为 {'node1': {'neighbor1', 'neighbor2'}, ...}
    """
    # 首先构建节点名称映射（标准化后的名称 -> 原始名称）
    node_mapping = {}
    for node in graph_a.nodes():
        normalized = normalize_init_node(node)
        node_mapping[normalized] = node

    # 添加边
    for source, targets in dict_b.items():
        # 标准化源节点名称
        normalized_source = normalize_init_node(source)

        # 如果图a中存在对应的源节点
        if normalized_source in node_mapping:
            actual_source = node_mapping[normalized_source]

            for target in targets:
                # 标准化目标节点名称
                normalized_target = normalize_init_node(target)

                # 如果图a中存在对应的目标节点
                if normalized_target in node_mapping:
                    actual_target = node_mapping[normalized_target]
                    # 添加边
                    if has_features(actual_source) or has_features(actual_target):
                        if not graph_a.has_edge(actual_source, actual_target):
                            graph_a.add_edge(actual_source, actual_target)

def has_features(node):
    return (len(node.permission) > 0 or
            len(node.sensitiveApi) > 0 or
            len(node.suspiciousApi) > 0 or
            len(node.url) > 0 or
            len(node.filterToken) > 0 or
            len(node.provider) > 0 or
            len(node.hardware_component) > 0)

def add_edges_from_dict(graph_a, dict_b):
    """
    将字典b中的边添加到图a中，不新增节点，处理<init>节点

    参数:
    graph_a: networkx图对象，节点类型为Node
    dict_b: 格式为 {'node1': {'neighbor1', 'neighbor2'}, ...}
    """
    # 首先构建节点名称映射（标准化后的名称 -> 原始Node对象）
    node_mapping = {}
    for node in graph_a.nodes():
        normalized = normalize_init_node(node)
        node_mapping[normalized] = node

    # 添加边
    for source, targets in dict_b.items():
        # 标准化源节点名称
        normalized_source = normalize_init_node(source)

        # 如果图a中存在对应的源节点
        if normalized_source in node_mapping:
            actual_source = node_mapping[normalized_source]

            for target in targets:
                # 标准化目标节点名称
                normalized_target = normalize_init_node(target)

                # 如果图a中存在对应的目标节点
                if normalized_target in node_mapping:
                    actual_target = node_mapping[normalized_target]

                    # 添加边
                    if not graph_a.has_edge(actual_source, actual_target):
                        graph_a.add_edge(actual_source, actual_target)

def clear_android_node(graph):
    base_packages = {
        "Landroid/",
        "Landroidx/",
        "Lcom/android/",
        "Ljava/",
        "Ljavax/",
        "Lsun/",
        "Lorg/xml/",
        "Lorg/json/",
        "Lorg/w3c/",
        "Lorg/apache/harmony/",
        "Lcom/google/android/"
    }

    # 收集需要删除的节点
    nodes_to_remove = set()

    # 遍历图中的所有节点
    for node in list(graph.nodes()):
        # 检查节点名称是否以任何基础包前缀开头
        if any(node.name.startswith(pkg) for pkg in base_packages):
            nodes_to_remove.add(node)
    graph.remove_nodes_from(nodes_to_remove)
    return graph

def get_label_llm(explainer,seq, feature_names,key_features,features_test,idx, label_llm_5, feature_num, llm_name,recheck,device,mali_p):
    has_check=0
    recheck=int(recheck)
    lines=0
    try:
        smali_tmp=r'.\decompiled_java\smali_tmp'
        baksmali_path = 'D:\\dexCompile\\program\\baksmali-2.5.2.jar'
        jadx_path = r"D:\jadx\bin\jadx.bat"
        smali_path=r'D:\\dexCompile\\program\\smali-2.5.2.jar'
        graph_dir=r'.\decompiled_java\graph_tmp'
        android_sdk=r'D:/androidSDK/platforms'
        soot_jar=r'D:\GitHub-download-package\CallGraph-Flowdroid-master\soot-infoflow-cmd-jar-with-dependencies.jar'
        if system == "Linux":
            soot_jar=r'/home/changxiaosong/python/malwareTest/pr2/soot-infoflow-cmd-jar-with-dependencies.jar:/home/changxiaosong/python/malwareTest/pr2_new/soot-4.2.1.jar:/home/changxiaosong/python/malwareTest/pr2_new/commons-io-2.6.jar:/home/changxiaosong/python/malwareTest/pr2_new/soot-infoflow-android-2.12.0.jar:/home/changxiaosong/python/malwareTest/pr2_new/soot-infoflow-2.12.0.jar:/home/changxiaosong/python/malwareTest/pr2_new/xmlpull-1.1.3.4d_b4_min.jar'
            android_sdk='/home/changxiaosong/python/obfuscapk/platforms'
            smali_tmp='/home/changxiaosong/python/malwareTest/pr2'+os.sep+'decompiled_java'+os.sep+'smali_tmp'
            baksmali_path = '/home/changxiaosong/dexCompile/program/baksmali-2.5.2.jar'
            jadx_path = r"/home/changxiaosong/jadx/bin/jadx"
            smali_path=r'/home/changxiaosong/python/malwareTest/smali-2.5.2.jar'
            graph_dir='/home/changxiaosong/python/malwareTest/pr2'+os.sep+'decompiled_java'+os.sep+'graph_tmp'
        apkName,apkPath=get_file_path_by_seq(seq)
        if system == "Linux":
            apkPath=apkPath.replace('F:/malware-app','/home/changxiaosong/dataset')
        else:
            apkPath=apkPath.replace('F:/malware-app','E:/malware-app')
        ApkFile = os.path.abspath(apkPath)


        #构建图
        graph_file=graph_dir+os.sep+str(seq)+'.pkl'
        if os.path.exists(graph_file):
            graph=load_graph(graph_file)
        else:
            #得到图
            if system == "Linux":
                thread_num=32
            else:
                thread_num=1
            #得到分析工具
            smali_dir,smali_status=get_smali(smali_tmp,baksmali_path,apkPath,apkName)
            try:
                smaliAnalyzer = SmaliAnalyzer(smali_dir)
                apkAnalyzer=ApkAnalyzer(ApkFile,smali_dir)
            except Exception as e:
                print('代码行',inspect.currentframe().f_lineno)
                print(e)
            per_api_dir=r'.'+os.sep+'sensitive_api_list'
            api2Permission=get_sensitive_apis_extend(per_api_dir)
            graph = generate_apk_method_graph(smali_dir, api2Permission, apkAnalyzer, smaliAnalyzer,thread_num)
            if graph is not None:
                save_graph(graph,graph_file)
        if graph is not None:
            graph_flowid=analyze_call_graph(android_sdk,soot_jar,seq,apkPath)
            #进行添加
            print(seq,'之前的边',len(graph.edges()))
            add_edges_from_dict(graph, graph_flowid)
            print(seq,'之后的边',len(graph.edges()))
            clear_nodes=[]
            processed_features = set()
            taskKernal_last=''
            while True:
                #移除可疑节点
                remove_nodes_preserve_edges(graph, clear_nodes)
                #得到特征
                x_sample_np = features_test[idx].toarray()
                # 将已处理的特征列清零
                if processed_features:
                    for feature_idx in processed_features:
                        x_sample_np[0, feature_idx] = 0
                x_sample_tensor = torch.tensor(x_sample_np, dtype=torch.float32).to(device)
                non_zero_shap_scores_sorted = get_risky_features(explainer, feature_names, x_sample_np, x_sample_tensor)
                key_features[seq]=non_zero_shap_scores_sorted

                feature_risky=[]
                all_feature=[]
                feature_value=[]
                current_processed_indices = set()

                key_features[seq] = sorted(non_zero_shap_scores_sorted,
                                           key=lambda x: abs(x[2]),  # 改为绝对值
                                           reverse=True)
                for name, feat_val, shap_val in key_features[seq][:feature_num]:
                    feature_risky.append(name)
                    feature_value.append(shap_val)
                    all_feature.append(f"{name} (Importance: {shap_val:.4f})")
                    # 记录本次处理的特征索引
                    feature_idx = np.where(feature_names == name)[0]
                    if len(feature_idx) > 0:
                        current_processed_indices.add(feature_idx[0])

                # 将本次处理的特征添加到已处理集合
                processed_features.update(current_processed_indices)
                #发现可疑节点
                result = search_matching_nodes(graph, feature_risky,feature_value)
                sorted_results = sorted(result.items(), key=lambda x: x[1], reverse=True)
                risk_nodes=[]
                for node, count in sorted_results:
                    if node.name == 'apk-base':
                        continue
                    risk_nodes.append(node)
                RM_str, AC_str, AI_str, CM_str =analyze_risk_components(seq, graph, risk_nodes,jadx_path,smali_path)

                RM_str, AC_str, AI_str, CM_str = clear(RM_str, AC_str, AI_str, CM_str)

                lines = RM_str.count('\n') + 1
                talks_1 = []
                talks_2 = []
                feature_prompt = build_feature_based_prompt(all_feature)
                function_Str=''
                functions={}
                if len(AC_str)> 0:
                    talks_1.append( {"role": "user", "content": f'''## Activate Methods:{AC_str}'''})
                if len(AI_str)> 0:
                    talks_1.append( {"role": "user", "content": f'''## Active Trace Methods:{AI_str}'''})
                if len(RM_str)> 0:
                    talks_1.append( {"role": "user", "content": f'''## Critical Methods:{RM_str}'''})
                if len(CM_str) > 0:
                    talks_1.append( {"role": "user", "content": f'''## Called Methods:{CM_str}'''})
                # 第一阶段对话
                if len(talks_1)>0:
                    talks_1.append( {"role": "user", "content": 'These code is from an APK file and will use ['+feature_prompt+']. While the user is utilizing [Activate Methods], the program activates [Critical Methods] through [Critical Methods], which in turn invokes [Called Methods]'})
                    for one in all_feature:
                        task = f'''Analyze how the feature **[{one}]** is actually used in this code.
1. **When/How is it triggered?** (e.g., on boot, automatically in background)
2. **What specific action does it perform?** (e.g., sends SMS to number X, reads IMSI and sends to server)
3. **What is the likely purpose?** (e.g., data theft, premium SMS fraud, persistence)'''
                        _, initial_reasoning  = llm_chat(0, talks_1.copy(), task, llm_name)
                        task = f'''Verify this analysis of [{one}] usage: "{initial_reasoning}"
                    Check: 1) Is it based on actual code?
                    Output ONLY a short, concise corrected version in ONE paragraph (max 3 sentences). Do not include explanations, examples, or additional text.'''
                        talks_1, function = llm_chat(0, talks_1, task, llm_name)

                        functions[one]=function
                for one in all_feature:
                    function_Str=function_Str+f"""\r\n{one} is used for {functions[one]}"""
                # 第二阶段对话
                task = f"""
As a malware analysis expert, make a final judgment by analyzing how features are actually used:

## Phase 1 Initial Result:
{'Malicious' if mali_p > 0.5 else 'Benign'}

## Key Features & Their Usage Context:
**Behavioral Features:** {str(feature_prompt)}
**Primary Function:** [{function_Str}]

## Core Analysis Tasks:
1. **Validate based on usage:** Does the feature usage match legitimate needs for [{function_Str}]?
2. **Identify abuse patterns:** How could these features be abused in real scenarios?
3. **Key malicious indicators:** Which features are most suspicious given their actual application?

## Output Format:
Final Classification: [Benign/Malicious]
Key Evidence: [1-2 most telling features and how they're being used]
Malicious Pattern: [e.g., mining, data theft - based on usage context]
Reasoning: [Brief: legitimate use vs actual usage mismatch]
Confidence: [High/Medium/Low]
"""
                talks_2, llm_ret = llm_chat(seq, talks_2, task, llm_name)
                label_llm_5 = get_label_loop(llm_ret,llm_name)
                prompt_dir = r'.' + os.sep + 'decompiled_java' + os.sep + 'prompt_dir_4'
                with open(prompt_dir + os.sep + str(seq) + ".txt", 'w') as f:
                    f.write(str(talks_1))
                    f.write('\r\n')
                    f.write(str(talks_2))
                    f.write('\r\n')
                    f.write(str(llm_ret))
                    f.write('\r\n')
                    f.write(str(label_llm_5))
                has_check+=1
                if int(label_llm_5)==0:
                    clear_nodes.extend([key for key in result.keys()])
                else:
                    if recheck-has_check>0:
                        print(seq,'剩余 '+str(recheck-has_check)+'次没有询问')
                    else:
                        print(seq,'询问完成')
                        break
    except Exception as e:
        traceback.print_exc()
    if isinstance(label_llm_5, np.int64):
        label_llm_5 = int(label_llm_5)
    return  label_llm_5,lines

lock_a = threading.Lock()
def get_risky_features(explainer, feature_names, x_sample_np, x_sample_tensor):
    with lock_a:
        if hasattr(x_sample_np, 'toarray'):
            x_sample_dense = x_sample_np.toarray()
        else:
            x_sample_dense = x_sample_np

        # 计算SHAP值
        shap_values = explainer.shap_values(x_sample_dense)

        sample_dense_vector = x_sample_dense[0]
        non_zero_indices = np.where(sample_dense_vector > 0)[0]
        non_zero_feature_names = feature_names[non_zero_indices]
        non_zero_feature_values = sample_dense_vector[non_zero_indices]

        # 对于二分类，shap_values[1]是正类的SHAP值
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]  # 二分类情况
        else:
            shap_vals = shap_values[0]     # 单输出情况

        non_zero_shap_values = shap_vals[non_zero_indices]
        non_zero_shap_scores = list(zip(non_zero_feature_names,
                                        non_zero_feature_values,
                                        non_zero_shap_values))
        non_zero_shap_scores_sorted = sorted(non_zero_shap_scores,
                                             key=lambda x: abs(x[2]),
                                             reverse=True)
        return non_zero_shap_scores_sorted


def remove_nodes_preserve_edges(graph, clear_nodes):
    """极简版本"""
    clear_names = {n.name for n in clear_nodes}

    # 构建重新连接映射
    for node in clear_nodes:
        if node in graph:
            parents = list(graph.predecessors(node))
            children = list(graph.successors(node))

            # 重新连接父节点到子节点
            for parent in parents:
                for child in children:
                    graph.add_edge(parent, child)
    # 删除节点和相关的边
    graph.remove_nodes_from(clear_nodes)

    return graph
import shutil, os

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def smali_2_java(jadx_path, seq, smali_path,smali_contents):
    seq=str(seq)
    if len(smali_contents)==0:
        return []
    base_dir= "." + os.sep + "decompiled_java" + os.sep + "output_tmp"+ os.sep +seq
    smali_dir = base_dir+ os.sep+'smali_little'
    java_path = base_dir+ os.sep+'java_little'
    clear_folder(smali_dir)
    clear_folder(java_path)
    i=0
    for smali_content in smali_contents:
        random_name = str(seq)+'-'+str(i)
        smali_file = smali_dir+os.sep + random_name + '.smali'
        with open(smali_file, 'w', encoding='utf-8') as file:
            file.write(smali_content)
        i+=1
    decompiler = SmaliDecompiler(smali_path, jadx_path)
    decompiler.decompile_smali_directory(
        smali_dir,
        java_path
    )
    content = ''
    # 读取并删除Java文件
    for root, dirs, files in os.walk(java_path):
        for filename in files:
            if filename.endswith(".java"):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    methods = f.readlines()
                    for one_line in methods:
                        content += one_line
                os.remove(file_path)
    clear_folder(smali_dir)
    clear_folder(java_path)
    return content

import platform
import sys
import re
from typing import Dict, List, Tuple

def convert_to_smali_format(java_method: str) -> str:
    """
    将Java格式的方法签名转换为Smali格式

    Args:
        java_method: Java格式的方法签名，如 "<dummyMainClass: void dummyMainMethod(java.lang.String[])>"

    Returns:
        Smali格式的方法签名
    """
    try:
        # 移除尖括号
        method = java_method.strip('<>')

        # 处理构造函数和静态初始化块
        if ' <init>' in method or ' <clinit>' in method:
            return convert_special_method(method)

        # 解析类名、返回类型、方法名和参数
        pattern = r'(.+):\s*([^\s]+)\s+([^\s(]+)\(([^)]*)\)'
        match = re.match(pattern, method)

        if not match:
            # 如果正则匹配失败，返回原始格式
            return f"L{method.replace('.', '/').replace(':', ';->')}"

        class_name, return_type, method_name, params = match.groups()

        # 转换类名为Smali格式
        smali_class = convert_class_to_smali(class_name)

        # 转换返回类型
        smali_return_type = convert_type_to_smali(return_type)

        # 转换参数
        if params and params.strip():
            param_list = [param.strip() for param in params.split(',')]
            smali_params = ''.join([convert_type_to_smali(param) for param in param_list])
        else:
            smali_params = ''

        return f"{smali_class}->{method_name}({smali_params}){smali_return_type}"

    except Exception as e:
        # 如果转换失败，返回原始方法名
        return f"CONVERSION_ERROR: {java_method}"


def convert_class_to_smali(class_name: str) -> str:
    """
    转换类名为Smali格式
    """
    # 处理数组类
    if class_name.endswith('[]'):
        base_class = class_name[:-2]
        return f'[L{base_class.replace(".", "/")};'

    # 处理基本类型数组
    primitive_arrays = {
        'int[]': '[I',
        'boolean[]': '[Z',
        'byte[]': '[B',
        'char[]': '[C',
        'short[]': '[S',
        'long[]': '[J',
        'float[]': '[F',
        'double[]': '[D'
    }
    if class_name in primitive_arrays:
        return primitive_arrays[class_name]

    # 普通类
    if not class_name.startswith('L') or not class_name.endswith(';'):
        return f'L{class_name.replace(".", "/")};'

    return class_name


def get_detailed_analysis(file_path: str):
    """
    获取详细的分析报告
    """
    call_graph = parse_dot_file(file_path)

    print("\n" + "="*60)
    print("详细分析报告")
    print("="*60 + "\n")

    # 按组件类型分类统计
    activity_calls = []
    service_calls = []
    receiver_calls = []
    unity_calls = []
    facebook_calls = []
    other_calls = []

    for caller, callees in call_graph.items():
        smali_caller = convert_to_smali_format(caller)

        for callee in callees:
            smali_callee = convert_to_smali_format(callee)

            # 分类
            if 'Activity' in callee:
                activity_calls.append((smali_caller, smali_callee))
            elif 'Service' in callee:
                service_calls.append((smali_caller, smali_callee))
            elif 'Receiver' in callee:
                receiver_calls.append((smali_caller, smali_callee))
            elif 'unity' in callee.lower() or 'Unity' in callee:
                unity_calls.append((smali_caller, smali_callee))
            elif 'facebook' in callee.lower() or 'Facebook' in callee:
                facebook_calls.append((smali_caller, smali_callee))
            else:
                other_calls.append((smali_caller, smali_callee))

    # 输出分类结果
    categories = [
        ("Activity调用关系", activity_calls),
        ("Service调用关系", service_calls),
        ("BroadcastReceiver调用关系", receiver_calls),
        ("Unity相关调用", unity_calls),
        ("Facebook相关调用", facebook_calls),
        ("其他调用关系", other_calls)
    ]

    for category_name, calls in categories:
        if calls:
            print(f"\n{category_name} ({len(calls)}个):")
            # 去重并排序
            unique_calls = sorted(set(calls))
            for caller, callee in unique_calls:
                print(f"  {caller}")
                print(f"    └─ {callee}")


def normalize_init_node(node_name):
    """标准化包含<init>的节点名称，去除形参"""
    # 如果传入的是Node对象，获取其name属性
    if hasattr(node_name, 'name'):
        node_name = node_name.name

    if '-><init>' in node_name:
        # 分离类名和方法签名
        class_part, method_part = node_name.split('-><init>')
        # 只保留类名和<init>，去除形参
        return f"{class_part}-><init>()"
    return node_name

def add_edges_from_dict(graph_a, dict_b):
    """
    将字典b中的边添加到图a中，不新增节点，处理<init>节点

    参数:
    graph_a: networkx图对象，节点类型为Node
    dict_b: 格式为 {'node1': {'neighbor1', 'neighbor2'}, ...}
    """
    # 首先构建节点名称映射（标准化后的名称 -> 原始Node对象）
    node_mapping = {}
    for node in graph_a.nodes():
        normalized = normalize_init_node(node)
        node_mapping[normalized] = node

    # 添加边
    for source, targets in dict_b.items():
        # 标准化源节点名称
        normalized_source = normalize_init_node(source)

        # 如果图a中存在对应的源节点
        if normalized_source in node_mapping:
            actual_source = node_mapping[normalized_source]

            for target in targets:
                # 标准化目标节点名称
                normalized_target = normalize_init_node(target)

                # 如果图a中存在对应的目标节点
                if normalized_target in node_mapping:
                    actual_target = node_mapping[normalized_target]

                    # 添加边
                    if not graph_a.has_edge(actual_source, actual_target):
                        graph_a.add_edge(actual_source, actual_target)

import pandas as pd
from typing import List, Dict, Optional

class ExcelDataLoader:
    def __init__(self, file_path: str):
        """
        初始化数据加载器
        :param file_path: Excel文件路径
        """
        self.file_path = file_path
        self.data_list = []  # 存储所有数据的列表
        self.seq_dict = {}   # 用于快速检索的字典，key为seq，value为对应行数据
        self.load_data()
    def load_data(self) -> List[Dict]:
        """
        加载Excel文件数据到列表和字典
        :return: 数据列表
        """
        try:
            # 读取Excel文件
            df = pd.read_csv(self.file_path)

            # 将DataFrame转换为字典列表
            self.data_list = df.to_dict('records')

            # 构建seq到数据的映射字典，用于快速检索
            self.seq_dict = {item['seq']: item for item in self.data_list if pd.notna(item.get('seq'))}

            print(f"成功加载 {len(self.data_list)} 条数据")
            return self.data_list

        except Exception as e:
            print(f"加载文件失败: {e}")
            return []

    def get_label_guess_by_seq(self, seq: int) -> Optional[int]:
        """
        根据seq检索并返回label_guess
        :param seq: 要检索的序列号
        :return: label_guess值，如果未找到返回None
        """
        if seq in self.seq_dict:
            item = self.seq_dict[seq]
            label_value = item.get('label_guess')
            if pd.isna(label_value) or label_value is None:
                return None
            return int(label_value)
        else:
            print(f"未找到seq为 {seq} 的数据")
            return None

    def get_full_data_by_seq(self, seq: int) -> Optional[Dict]:
        """
        根据seq检索并返回完整数据
        :param seq: 要检索的序列号
        :return: 完整数据字典，如果未找到返回None
        """
        if seq in self.seq_dict:
            return self.seq_dict[seq]
        else:
            print(f"未找到seq为 {seq} 的数据")
            return None

    def get_all_data(self) -> List[Dict]:
        """
        获取所有数据
        :return: 所有数据的列表
        """
        return self.data_list
