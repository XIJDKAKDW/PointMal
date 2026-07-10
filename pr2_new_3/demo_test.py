#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：malwareTest
@File    ：gradient_boosting_experiment.py
@IDE     ：PyCharm
@Author  ：常晓松
@Date    ：2025/9/5 9:45
"""
import json
import multiprocessing as mp
import os
import pickle
import platform
import sys
import warnings
from typing import List, Dict
import uuid
import numpy as np
import pandas as pd
import pymysql
import shap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

system = platform.system()

if system == "Linux":
    sys.path.append(r"/home/changxiaosong/python/malwareTest")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr2_new_3")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr2_new_2")

from pr2_new_3.test001Method_new_9_4_3 import llm_chat
def get_label_loop(llm_ret):
    label=''
    for one in llm_ret.split('\n'):
        if 'Final Verdict' in one:
            label=one
            break
    return 1 if 'Malicious' in label else 0
warnings.filterwarnings('ignore')

# 设置随机种子
SEED = 42
np.random.seed(SEED)


def get_feature_file_label_by_seq(conn, seqs):
    """根据序列号获取文件路径和标签"""
    file_paths = []
    labels = []

    for seq in seqs:
        try:
            with conn.cursor() as cursor:
                # 获取文件路径
                sql = "SELECT path FROM drebin_feature WHERE apkSeq = %s"
                cursor.execute(sql, (seq,))
                result = cursor.fetchone()

                if result and result[0]:
                    file_path = result[0]
                    if platform.system() != "Linux":
                        file_path = file_path.replace('/home/changxiaosong/dataset', r'D:')
                    file_paths.append(file_path)

                    # 获取标签
                    sql = "SELECT label FROM app_label WHERE seq = %s"
                    cursor.execute(sql, (seq,))
                    label_result = cursor.fetchone()
                    if label_result:
                        labels.append(0 if label_result[0] == 'B' else 1)
                    else:
                        labels.append(0)
        except Exception as e:
            print(f"获取序列 {seq} 的文件路径和标签失败: {e}")
            continue

    return file_paths, labels


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
        port=3306,
    )
    return conn


def load_single_graph_file(file_label_pair):
    """加载单个图文件的函数（必须放在顶层才能被pickle）"""
    graph_file, label = file_label_pair
    try:
        with open(graph_file, 'rb') as f:
            graph = pickle.load(f)
            if graph is not None and len(graph.nodes()) > 0:
                return graph, label
    except Exception as e:
        print(f"加载图文件 {graph_file} 失败: {e}")
    return None, None


class OptimizedGraphFeatureExtractor:
    """优化的图特征提取器"""

    def __init__(self):
        self.node_feature_names = []
        self.edge_feature_names = []
        self.feature_cache = {}  # 添加特征缓存

    def extract_node_features_fast(self, graph) -> Dict[str, str]:
        """快速提取节点级别的特征"""
        node_features = {}

        # 批量处理节点
        nodes = list(graph.nodes())
        for node in nodes:
            node_name = node.name
            if node_name == 'apk-base':
                continue
            features = []

            # 权限特征
            if hasattr(node, 'permission'):
                permissions = node.permission if isinstance(node.permission, list) else [node.permission]
                for perm in permissions:
                    if perm:
                        feature_name = perm
                        features.append(feature_name)
                        if feature_name not in self.node_feature_names:
                            self.node_feature_names.append(feature_name)

            # 敏感API特征
            if hasattr(node, 'sensitiveApi'):
                apis = node.sensitiveApi if isinstance(node.sensitiveApi, list) else [node.sensitiveApi]
                for api in apis:
                    if api:
                        feature_name = api
                        features.append(feature_name)
                        if feature_name not in self.node_feature_names:
                            self.node_feature_names.append(feature_name)

            # 可疑API特征
            if hasattr(node, 'suspiciousApi'):
                susp_apis = node.suspiciousApi if isinstance(node.suspiciousApi, list) else [node.suspiciousApi]
                for api in susp_apis:
                    if api:
                        feature_name = api
                        features.append(feature_name)
                        if feature_name not in self.node_feature_names:
                            self.node_feature_names.append(feature_name)

            # URL特征
            if hasattr(node, 'url'):
                urls = node.url if isinstance(node.url, list) else [node.url]
                for url in urls:
                    if url:
                        feature_name = url
                        features.append(feature_name)
                        if feature_name not in self.node_feature_names:
                            self.node_feature_names.append(feature_name)

            # 组件特征
            if hasattr(node, 'component'):
                if node.component:
                    feature_name = node.component
                    features.append(feature_name)
                    if feature_name not in self.node_feature_names:
                        self.node_feature_names.append(feature_name)

            node_features[node_name] = ' '.join(features)

        return node_features

    def extract_global_features_fast(self, graph) -> str:
        """快速提取全局图特征"""
        # 尝试从缓存中获取

        random_suffix = str(uuid.uuid4())[:8]  # 取前8位
        graph_id = f"{id(graph)}_{random_suffix}"
        if graph_id in self.feature_cache:
            return self.feature_cache[graph_id]

        node_features = self.extract_node_features_fast(graph)

        # 将所有节点特征合并
        all_features = []
        for features in node_features.values():
            if features.strip():
                all_features.append(features)

        result = ' '.join(all_features)

        # 缓存结果
        self.feature_cache[graph_id] = result
        return result


def extract_single_global_features_wrapper(args):
    """包装函数用于多进程处理"""
    extractor, graph = args
    return extractor.extract_global_features_fast(graph)



class WeightedTfidfVectorizer:
    """带恶意性权重的TF-IDF向量化器"""

    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.feature_weights = {}
        self.malicious_count = {}
        self.benign_count = {}
        self.total_samples = 0

    def fit(self, X, y):
        """训练向量化器，计算恶意性权重"""
        # 先计算每个特征在恶意和良性样本中的出现次数
        self.total_samples = len(X)

        # 初始化计数
        all_features = set()
        for text in X:
            features = set(text.split())
            all_features.update(features)

        # 初始化计数字典
        for feature in all_features:
            self.malicious_count[feature] = 0
            self.benign_count[feature] = 0

        # 统计每个特征在不同类别中的出现次数
        for text, label in zip(X, y):
            features = set(text.split())
            for feature in features:
                if label == 1:  # 恶意样本
                    self.malicious_count[feature] = self.malicious_count.get(feature, 0) + 1
                else:  # 良性样本
                    self.benign_count[feature] = self.benign_count.get(feature, 0) + 1

        # 计算恶意性权重
        for feature in all_features:
            mal_count = self.malicious_count.get(feature, 0)
            ben_count = self.benign_count.get(feature, 0)

            if mal_count + ben_count > 0:
                # 权重公式：恶意出现比例 * log(总出现次数+1)
                mal_ratio = mal_count / (mal_count + ben_count) if (mal_count + ben_count) > 0 else 0
                total_count = mal_count + ben_count
                weight = mal_ratio * np.log(total_count + 1)
                self.feature_weights[feature] = weight
            else:
                self.feature_weights[feature] = 0.0

        # 使用原始TF-IDF进行训练
        self.vectorizer.fit(X)

        return self

    def transform(self, X):
        """转换特征，应用恶意性权重"""
        # 获取原始TF-IDF特征
        tfidf_features = self.vectorizer.transform(X)

        # 转换为稠密矩阵以便操作
        tfidf_dense = tfidf_features.toarray()

        # 获取特征名称
        feature_names = self.vectorizer.get_feature_names_out()

        # 创建权重向量
        weights = np.ones(len(feature_names))
        for i, feature in enumerate(feature_names):
            weights[i] = self.feature_weights.get(feature, 1.0)

        # 应用权重
        weighted_features = tfidf_dense * weights

        return weighted_features

    def fit_transform(self, X, y):
        """训练并转换特征"""
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names(self):
        """获取特征名称"""
        return self.vectorizer.get_feature_names_out()


class CriticalFeatureAnalyzer:
    """关键特征分析器，用于定位恶意区域并调用LLM分析"""

    def __init__(self, llm_model_name='deepseek-coder-v2:16b'):
        self.llm_model_name = llm_model_name
        self.explainer = None
        self.feature_names = None

    def set_shap_explainer(self, model, X_train):
        """设置SHAP解释器"""
        # 创建SHAP TreeExplainer
        self.explainer = shap.TreeExplainer(model)

    def analyze_critical_features(self, graph, features_vector, feature_names, model, seq,
                                  threshold=0.1, top_k=5):
        """分析关键特征，定位恶意区域"""
        if self.explainer is None:
            raise ValueError("SHAP解释器未设置")

        # 获取样本的SHAP值
        shap_values = self.explainer.shap_values(features_vector)

        # 对于二分类问题，取正类的SHAP值
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]  # 恶意类别的SHAP值
        else:
            shap_vals = shap_values

        # 获取非零特征及其SHAP值
        non_zero_indices = np.where(features_vector[0] > 0)[0]
        feature_contributions = []

        for idx in non_zero_indices:
            feature_name = feature_names[idx]
            shap_value = shap_vals[0][idx]
            feature_value = features_vector[0][idx]
            feature_contributions.append((feature_name, feature_value, shap_value))

        # 按SHAP值绝对值排序
        feature_contributions.sort(key=lambda x: abs(x[2]), reverse=True)

        # 取最重要的top_k个特征
        top_features = feature_contributions[:top_k]

        # 在图中定位这些特征对应的节点
        critical_nodes = self.locate_features_in_graph(graph, top_features)

        # 提取节点代码并调用LLM分析
        analysis_result = self.analyze_critical_nodes_with_llm(critical_nodes, top_features, seq)

        return {
            'top_features': top_features,
            'critical_nodes': critical_nodes,
            'llm_analysis': analysis_result,
            'seq': seq
        }

    def locate_features_in_graph(self, graph, top_features):
        """在图中定位特征对应的节点"""
        critical_nodes = {}

        for feature_name, feature_value, shap_value in top_features:

            # 在图中搜索匹配该特征的节点
            matching_nodes = self.search_nodes_by_feature(graph,feature_name,feature_value)

            if matching_nodes:
                critical_nodes[feature_name] = {
                    'nodes': matching_nodes,
                    'shap_value': shap_value,
                    'feature_value': feature_value
                }

        return critical_nodes


    def search_nodes_by_feature(self,graph, feature_name,feature_value):
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
                matching_nodes.append({
                    'node_name': node.name,
                    'feature_value': feature_value,
                    'node_object': node
                })
        return matching_nodes
    def analyze_critical_nodes_with_llm(self, critical_nodes, top_features, seq):
        """使用LLM分析关键节点"""
        if not critical_nodes:
            return {"error": "未找到关键节点"}

        # 构建分析提示
        prompt = self.build_analysis_prompt(critical_nodes, top_features, seq)

        llm_response = self.call_llm_api(prompt)
        return self.parse_llm_response(llm_response)

    def build_analysis_prompt(self, critical_nodes, top_features, seq):
        """Build LLM analysis prompt"""
        prompt_parts = [
            "You are an Android malware analysis expert. Please analyze the suspicious code regions in the following APK sample.",
            f"Sample sequence number: {seq}",
            "",
            "## Key Feature Analysis:"
        ]

        # Add feature information
        for i, (feature_name, feature_value, shap_value) in enumerate(top_features, 1):
            prompt_parts.append(f"{i}. Feature: {feature_name}")
            prompt_parts.append(f"   SHAP value: {shap_value:.4f} (impact level)")
            if feature_name in critical_nodes:
                nodes = critical_nodes[feature_name]['nodes']
                prompt_parts.append(f"   Number of nodes involved: {len(nodes)}")

        prompt_parts.append("")
        prompt_parts.append("## Suspicious Node Code Regions:")

        # Add node information
        node_counter = 1
        for feature_name, feature_data in critical_nodes.items():
            nodes = feature_data['nodes']
            for node_info in nodes[:3]:  # Limit to max 3 nodes per feature
                node = node_info['node_object']
                prompt_parts.append(f"\n{node_counter}. Node: {node_info['node_name']}")
                prompt_parts.append(f"   Associated feature: {feature_name}")

                # Add node attributes
                if hasattr(node, 'body') and node.body:
                    # Extract code snippet (first 20 lines)
                    lines = node.body.split('\n')[:20]
                    code_snippet = '\n'.join(lines)
                    prompt_parts.append(f"   Code snippet:\n```smali\n{code_snippet}\n```")

                node_counter += 1

        prompt_parts.append("")
        prompt_parts.append("## Analysis Tasks:")
        prompt_parts.append("1. Evaluate whether these code regions truly contain malicious behavior")
        prompt_parts.append("2. Analyze whether the feature combinations form malicious patterns")
        prompt_parts.append("3. Determine if this is a false positive (normal functionality incorrectly flagged)")
        prompt_parts.append("4. Provide final classification recommendation and confidence level")
        prompt_parts.append("")
        prompt_parts.append("## Output Format:")
        prompt_parts.append("Final verdict: [Malicious/Benign/Uncertain]")
        prompt_parts.append("Confidence level: [High/Medium/Low]")
        prompt_parts.append("Key evidence: [List critical evidence]")
        prompt_parts.append("False positive possibility: [High/Medium/Low]")
        prompt_parts.append("Detailed analysis: [Detailed analysis process]")

        return '\n'.join(prompt_parts)

    def call_llm_api(self, prompt):
        try:
            talks_2, llm_ret = llm_chat(0, [], prompt, 'deepseek-coder-v2:16b')
            return llm_ret
        except Exception as e:
            print(f"LLM调用失败: {e}")
            return "无法获取LLM分析结果"

    def parse_llm_response(self, response):
        """解析LLM响应"""
        result = {
            'final_judgment': 'unknown',
            'confidence': 'medium',
            'key_evidence': [],
            'false_positive_likelihood': 'medium',
            'detailed_analysis': response
        }
        label_llm_5 = get_label_loop(response)
        result['final_judgment'] = 'Malicious' if label_llm_5 >= 0.5 else 'Benign'
        print('cxs1',response)
        return result


class FalsePositiveCorrector:
    """误报矫正器"""

    def __init__(self, critical_analyzer):
        self.critical_analyzer = critical_analyzer
        self.correction_history = []

    def correct_false_positives(self, predictions, graphs, feature_vectors, feature_names, model, seqs):
        """矫正误报预测"""
        corrected_predictions = predictions.copy()
        correction_details = []

        for i, (pred, graph, features, seq) in enumerate(zip(predictions, graphs, feature_vectors, seqs)):
            # 只处理预测为恶性的样本（可能的误报）
            if pred == 1:
                # 分析关键特征
                analysis = self.critical_analyzer.analyze_critical_features(
                    graph, features.reshape(1, -1), feature_names, model, seq
                )
                # 根据LLM分析结果决定是否矫正
                llm_judgment = analysis['llm_analysis']['final_judgment']
                false_positive_likelihood = analysis['llm_analysis']['false_positive_likelihood']

                # 如果LLM判断为良性或误报可能性高，则矫正预测
                if 'Benign' in llm_judgment or false_positive_likelihood == '高':
                    corrected_predictions[i] = 0
                    correction_details.append({
                        'seq': seq,
                        'original_prediction': 1,
                        'corrected_prediction': 0,
                        'llm_analysis': analysis['llm_analysis'],
                        'top_features': analysis['top_features']
                    })
                    print(f"矫正误报: seq={seq}, 理由: {analysis['llm_analysis']['detailed_analysis']}")

        return corrected_predictions, correction_details


def load_graphs_from_files_fast(graph_files, labels, num_workers=4):
    """快速从文件加载图（并行）"""
    graphs = []
    valid_labels = []

    print(f"使用 {num_workers} 个进程并行加载图文件...")

    # 准备数据对
    file_label_pairs = list(zip(graph_files, labels))

    # 并行加载
    if num_workers <= 1:
        # 单进程
        for file_label_pair in tqdm(file_label_pairs, desc="单进程加载图数据"):
            graph, label = load_single_graph_file(file_label_pair)
            if graph is not None:
                graphs.append(graph)
                valid_labels.append(label)
    else:
        # 多进程
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(load_single_graph_file, file_label_pairs),
                total=len(file_label_pairs),
                desc="并行加载图数据"
            ))

        # 处理结果
        for graph, label in results:
            if graph is not None:
                graphs.append(graph)
                valid_labels.append(label)

    return graphs, valid_labels


def extract_global_features_batch_fast(graphs, feature_extractor, num_workers=4):
    """快速批量提取全局特征"""
    print(f"使用 {num_workers} 个进程并行提取全局特征...")
    # 多进程 - 准备参数
    task_args = [(feature_extractor, graph) for graph in graphs]

    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(extract_single_global_features_wrapper, task_args),
            total=len(task_args),
            desc="并行提取全局特征"
        ))

    return results


def train_gradient_boosting(X_train, y_train, X_val, y_val):
    """训练梯度提升模型"""
    print("开始训练梯度提升模型...")

    # 使用指定的GradientBoostingClassifier参数
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=SEED
    )

    # 训练模型
    model.fit(X_train, y_train)

    # 验证集评估
    y_val_pred = model.predict(X_val)
    val_acc = balanced_accuracy_score(y_val, y_val_pred)

    print(f"验证集平衡准确率: {val_acc:.4f}")

    return model


def evaluate_gradient_boosting(model, X_test, y_test):
    """评估梯度提升模型性能"""
    print("评估模型性能...")

    # 预测
    y_pred = model.predict(X_test)

    # 计算评估指标
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    return balanced_acc, precision, recall, f1, y_pred, y_test, cm


def load_seqs_from_file(file_path: str) -> List[int]:
    """从文件加载序列号"""
    seqs = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and line.isdigit():
                    seqs.append(int(line))
    except Exception as e:
        print(f"加载文件时出错: {e}")
    return seqs


def run_false_positive_correction(predictions, test_graphs, X_test_tfidf, feature_names,
                                  model, seqs_test, llm_model_name='deepseek-coder-v2:16b'):
    # 初始化关键特征分析器
    critical_analyzer = CriticalFeatureAnalyzer(llm_model_name=llm_model_name)
    critical_analyzer.set_shap_explainer(model, X_test_tfidf)
    critical_analyzer.feature_names = feature_names

    # 初始化误报矫正器
    corrector = FalsePositiveCorrector(critical_analyzer)

    # 执行误报矫正
    corrected_predictions, correction_details = corrector.correct_false_positives(
        predictions, test_graphs, X_test_tfidf, feature_names, model, seqs_test
    )

    # 保存矫正详情
    if correction_details:
        with open('false_positive_corrections.json', 'w', encoding='utf-8') as f:
            json.dump(correction_details, f, indent=2, ensure_ascii=False)
        print(f"已保存 {len(correction_details)} 个矫正记录到 false_positive_corrections.json")

    return corrected_predictions, correction_details


def main():
    """主实验流程"""
    # 检查模型文件是否存在
    model_filename = 'gradient_boosting_model.pkl'
    feature_extractor_filename = 'feature_extractor.pkl'
    tfidf_filename = 'weighted_tfidf_vectorizer.pkl'

    # 检查所有必要的模型文件是否存在
    models_exist = (os.path.exists(model_filename) and
                    os.path.exists(feature_extractor_filename) and
                    os.path.exists(tfidf_filename))

    # 1. 加载测试集（训练和推理都需要）
    print("=" * 60)
    print("步骤1: 加载测试集的图")
    print("=" * 60)

    test_file = r'/home/changxiaosong/python/malwareTest/test_0.8repartition.txt'
    train_file = r'/home/changxiaosong/python/malwareTest/train_0.8repartition.txt'
    graph_dir = '/home/changxiaosong/python/malwareTest/pr2' + os.sep + \
                'decompiled_java' + os.sep + 'graph_tmp' + os.sep + 'gra_old'
    seqs_test = load_seqs_from_file(test_file)
    # seqs_test = [115723,114552,95178,9036,151821]

    # 设置并行工作进程数
    num_workers = min(mp.cpu_count(), 64)

    if models_exist:
        print("检测到已训练的模型，直接加载使用...")
        # 2. 加载模型和特征提取器
        print("\n" + "=" * 60)
        print("步骤2: 加载预训练的模型和特征提取器")
        print("=" * 60)

        import joblib
        model = joblib.load(model_filename)
        print(f"已加载模型: {model_filename}")

        with open(feature_extractor_filename, 'rb') as f:
            feature_extractor = pickle.load(f)
        print(f"已加载特征提取器: {feature_extractor_filename}")

        with open(tfidf_filename, 'rb') as f:
            weighted_tfidf = pickle.load(f)
        print(f"已加载TF-IDF向量化器: {tfidf_filename}")
    else:
        print("未检测到完整模型文件，执行完整训练流程...")
        # 加载训练集
        print("\n" + "=" * 60)
        print("步骤1a: 加载训练集的图")
        print("=" * 60)
        seqs_train = load_seqs_from_file(train_file)
        # seqs_train = [115723,114552,95178,9036,151821]
        conn = get_connection()
        _, train_labels = get_feature_file_label_by_seq(conn, seqs_train)
        conn.close()

        train_graph_files = []
        for seq in seqs_train:
            graph_file = graph_dir + os.sep + str(seq) + '.pkl'
            train_graph_files.append(graph_file)

        # 并行加载训练和测试集图数据
        train_graphs, train_labels = load_graphs_from_files_fast(
            train_graph_files, train_labels, num_workers=num_workers
        )

        print(f"成功加载训练集: {len(train_graphs)}个图")

        # 2. 提取全局特征
        print("\n" + "=" * 60)
        print("步骤2: 批量提取全局特征")
        print("=" * 60)

        feature_extractor = OptimizedGraphFeatureExtractor()

        # 并行提取特征
        train_global_features = extract_global_features_batch_fast(
            train_graphs, feature_extractor, num_workers=num_workers
        )

        print(f"训练集特征数量: {len(train_global_features)}")
        # 3. TF-IDF向量化（带恶意性权重）
        print("\n" + "=" * 60)
        print("步骤3: TF-IDF向量化（带恶意性权重）")
        print("=" * 60)

        weighted_tfidf = WeightedTfidfVectorizer(max_features=1000)
        X_train_tfidf = weighted_tfidf.fit_transform(train_global_features, train_labels)

        print(f"TF-IDF特征维度: {X_train_tfidf.shape[1]}")

        # 4. 划分训练集和验证集
        print("\n" + "=" * 60)
        print("步骤4: 划分训练集和验证集")
        print("=" * 60)

        # 随机划分训练集为训练和验证
        train_size = len(train_labels)
        indices = np.arange(train_size)
        np.random.shuffle(indices)

        split_point = int(0.8 * train_size)
        train_indices = indices[:split_point]
        val_indices = indices[split_point:]

        X_train = X_train_tfidf[train_indices]
        y_train = np.array(train_labels)[train_indices]

        X_val = X_train_tfidf[val_indices]
        y_val = np.array(train_labels)[val_indices]

        print(f"训练集大小: {len(X_train)}")
        print(f"验证集大小: {len(X_val)}")

        # 5. 训练梯度提升模型
        print("\n" + "=" * 60)
        print("步骤5: 训练梯度提升模型")
        print("=" * 60)

        model = train_gradient_boosting(X_train, y_train, X_val, y_val)

        # 保存模型和特征提取器
        print("\n保存模型和特征提取器...")

        import joblib
        joblib.dump(model, model_filename)
        print(f"模型已保存到 {model_filename}")

        with open(feature_extractor_filename, 'wb') as f:
            pickle.dump(feature_extractor, f)
        print(f"特征提取器已保存到 {feature_extractor_filename}")

        with open(tfidf_filename, 'wb') as f:
            pickle.dump(weighted_tfidf, f)
        print(f"TF-IDF向量化器已保存到 {tfidf_filename}")

    # 加载测试集
    conn = get_connection()
    _, test_labels = get_feature_file_label_by_seq(conn, seqs_test)
    conn.close()

    test_graph_files = []
    for seq in seqs_test:
        graph_file = graph_dir + os.sep + str(seq) + '.pkl'
        test_graph_files.append(graph_file)

    print(f"测试集: {len(test_graph_files)}个图")

    # 加载测试集图数据
    test_graphs, test_labels = load_graphs_from_files_fast(
        test_graph_files, test_labels, num_workers=num_workers
    )
    print(f"成功加载测试集: {len(test_graphs)}个图")

    # 提取测试集特征
    test_global_features = extract_global_features_batch_fast(
        test_graphs, feature_extractor, num_workers=num_workers
    )
    print(test_global_features)
    print(f"测试集特征数量: {len(test_global_features)}")

    # 转换特征
    X_test_tfidf = weighted_tfidf.transform(test_global_features)
    y_test = np.array(test_labels)

    print(f"测试集大小: {len(X_test_tfidf)}")

    # 6. 模型评估（训练和推理都需要）
    print("\n" + "=" * 60)
    print("步骤6: 模型评估")
    print("=" * 60)

    balanced_acc, precision, recall, f1, preds, labels, cm = evaluate_gradient_boosting(
        model, X_test_tfidf, y_test
    )

    # 输出基础评估结果
    print("\n" + "=" * 60)
    print("基础评估结果")
    print("=" * 60)
    print(f"平衡准确率 (Balanced Accuracy): {balanced_acc:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数: {f1:.4f}")

    print(f"\n混淆矩阵:")
    print(f"真阴性 (TN): {cm[0, 0]}")
    print(f"假阳性 (FP): {cm[0, 1]}")
    print(f"假阴性 (FN): {cm[1, 0]}")
    print(f"真阳性 (TP): {cm[1, 1]}")

    # 7. 误报矫正分析
    print("\n" + "=" * 60)
    print("步骤7: 误报矫正分析")
    print("=" * 60)

    # 获取特征名称
    feature_names = weighted_tfidf.get_feature_names()

    sample_size = len(seqs_test)
    sample_indices = np.random.choice(len(seqs_test), sample_size, replace=False)

    sample_preds = preds[sample_indices]
    sample_graphs = [test_graphs[i] for i in sample_indices]
    sample_features = X_test_tfidf[sample_indices]
    sample_seqs = [seqs_test[i] for i in sample_indices]
    print(f"\n矫正效果评估 (样本大小: {sample_size}):")
    # 运行误报矫正
    corrected_preds, correction_details = run_false_positive_correction(
        sample_preds, sample_graphs, sample_features, feature_names,
        model, sample_seqs
    )

    # 评估矫正效果
    sample_true_labels = y_test[sample_indices]

    # 计算矫正前后的指标
    original_balanced_acc = balanced_accuracy_score(sample_true_labels, sample_preds)
    corrected_balanced_acc = balanced_accuracy_score(sample_true_labels, corrected_preds)

    original_precision = precision_score(sample_true_labels, sample_preds, zero_division=0)
    corrected_precision = precision_score(sample_true_labels, corrected_preds, zero_division=0)

    original_recall = recall_score(sample_true_labels, sample_preds, zero_division=0)
    original_f1 = f1_score(sample_true_labels, sample_preds, zero_division=0)

    corrected_recall = recall_score(sample_true_labels, corrected_preds, zero_division=0)
    corrected_f1 = f1_score(sample_true_labels, corrected_preds, zero_division=0)

    print(f"\n矫正效果评估 (样本大小: {sample_size}):")
    print(f"矫正后平衡准确率: {corrected_balanced_acc:.4f}")
    print(f"提升: {corrected_balanced_acc - original_balanced_acc:+.4f}")

    print(f"矫正后精确率: {corrected_precision:.4f}")
    print(f"提升: {corrected_precision - original_precision:+.4f}")

    print(f"矫正后召回率: {corrected_recall:.4f}")
    print(f"提升: {corrected_recall - original_recall:+.4f}")

    print(f"矫正后F1: {corrected_f1:.4f}")
    print(f"提升: {corrected_f1 - original_f1:+.4f}")

    # 统计矫正情况
    corrections_count = len(correction_details)
    true_positives_corrected = 0
    false_positives_corrected = 0

    for detail in correction_details:
        seq = detail['seq']
        idx = sample_seqs.index(seq)
        true_label = sample_true_labels[idx]

        if true_label == 0:  # 原本是良性，被正确矫正
            false_positives_corrected += 1
        else:  # 原本是恶意，被错误矫正
            true_positives_corrected += 1

    print(f"\n矫正统计:")
    print(f"总矫正数: {corrections_count}")
    print(f"正确矫正的误报: {false_positives_corrected}")
    print(f"错误矫正的正报: {true_positives_corrected}")

    # 8. 保存完整结果
    print("\n" + "=" * 60)
    print("步骤8: 保存结果")
    print("=" * 60)

    # 准备完整结果
    results = {
        'model_type': 'GradientBoostingClassifier',
        'balanced_accuracy': float(balanced_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'seqs_test': seqs_test,
        'predictions': preds.tolist() if hasattr(preds, 'tolist') else list(preds),
        'true_labels': labels.tolist() if hasattr(labels, 'tolist') else list(labels),
        'test_set_size': len(X_test_tfidf),
        'mode': 'inference' if models_exist else 'training',
        'false_positive_correction': {
            'sample_size': sample_size,
            'corrected_balanced_acc': float(corrected_balanced_acc),
            'corrected_precision': float(corrected_precision),
            'corrected_recall': float(corrected_recall),
            'corrected_f1': float(corrected_f1),
        }
    }

    if not models_exist:
        results['model_params'] = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': SEED
        }
        results['feature_dimension'] = X_train_tfidf.shape[1]

    # 保存矫正详情
    if correction_details:
        results['correction_details'] = correction_details

    # 确定结果文件名
    if models_exist:
        results_filename = 'gradient_boosting_evaluation_results_inference.json'
    else:
        results_filename = 'gradient_boosting_evaluation_results.json'

    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n完整结果已保存到 {results_filename}")

    # 保存矫正详情到单独文件
    if correction_details:
        correction_filename = 'false_positive_correction_details.json'
        with open(correction_filename, 'w', encoding='utf-8') as f:
            json.dump({
                'total_corrections': len(correction_details),
                'corrections': correction_details,
                'timestamp': pd.Timestamp.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        print(f"详细矫正记录已保存到 {correction_filename}")

    print("\n" + "=" * 60)
    print("实验完成！")
    if models_exist:
        print("（推理模式 + 误报矫正）")
    else:
        print("（完整训练 + 误报矫正）")
    print("=" * 60)


if __name__ == "__main__":
    # 设置多进程启动方式
    if system == "Linux" and mp.get_start_method(allow_none=True) != 'forkserver':
        mp.set_start_method('forkserver', force=True)
    elif mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

    main()
