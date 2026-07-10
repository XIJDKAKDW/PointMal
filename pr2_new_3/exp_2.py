#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
脚本：使用预训练模型和对比学习模型分析多个函数
- 加载DeepSeek模型和训练好的对比学习模型
- 对输入函数进行向量化和嵌入
- 使用K-means聚类和边界距离计算恶意性分数
- 可视化多个函数在测试集分布中的位置
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
import os
import logging
from typing import List, Tuple, Optional, Dict, Any
import argparse
from collections import Counter

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# ==================== 配置 ====================
class Config:
    MODEL_PATH = r"/home/changxiaosong/python/malwareTest/deepseek-coder-1.3b-base"
    CONTRASTIVE_MODEL_PATH = "visualizations/contrastive_model.pth"
    TEST_EMBEDDINGS_PATH = "visualizations/test_contrastive_embeddings.npy"
    TEST_LABELS_PATH = "visualizations/test_labels.npy"
    RESULTS_DIR = "function_analysis"

    # 模型参数
    INPUT_DIM = 2048
    HIDDEN_DIM = 256
    OUTPUT_DIM = 128

    # 聚类参数
    N_CLUSTERS = 2  # 二分簇
    BOUNDARY_PERCENTILE = 90  # 使用90%分位数作为边界阈值


# ==================== 修复后的对比学习模型 ====================
class ContrastiveModelInference(nn.Module):
    """用于推理的对比学习模型"""
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return F.normalize(self.encoder(x), p=2, dim=1)


class OriginalContrastiveModel(nn.Module):
    """原始训练时的模型结构"""
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return F.normalize(self.encoder(x), p=2, dim=1)


# ==================== DeepSeek编码器 ====================
class DeepSeekEncoder:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        logger.info(f"加载DeepSeek模型到 {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32
        ).to(self.device)
        self.model.eval()
        logger.info("DeepSeek模型加载完成")

    def encode(self, code: str, max_length: int = 256) -> np.ndarray:
        """获取单个代码向量"""
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            embedding = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

        return embedding.cpu().numpy().squeeze()


# ==================== 改进后的距离分析器 ====================
class BoundaryDistanceAnalyzer:
    """基于聚类和边界距离的分析器"""

    def __init__(self, test_embeddings: np.ndarray, test_labels: List[int],
                 n_clusters: int = 2, boundary_percentile: float = 90):
        """
        初始化分析器

        Args:
            test_embeddings: 测试集嵌入
            test_labels: 测试集标签
            n_clusters: 聚类数量
            boundary_percentile: 边界阈值百分位数
        """
        self.test_embeddings = test_embeddings
        self.test_labels = np.array(test_labels)
        self.n_clusters = n_clusters
        self.boundary_percentile = boundary_percentile

        # 执行K-means聚类
        logger.info("执行K-means聚类...")
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(test_embeddings)

        # 计算每个簇的恶意百分比
        cluster_malware_ratios = {}
        self.cluster_majority_labels = {}
        self.cluster_boundary_thresholds = {}

        for cluster_id in range(n_clusters):
            cluster_mask = (self.cluster_labels == cluster_id)
            cluster_true_labels = self.test_labels[cluster_mask]

            if len(cluster_true_labels) > 0:
                # 计算恶意样本百分比
                malware_ratio = np.mean(cluster_true_labels == 1)
                cluster_malware_ratios[cluster_id] = malware_ratio

                # 计算到簇中心的距离分布
                cluster_center = self.kmeans.cluster_centers_[cluster_id]
                cluster_points = test_embeddings[cluster_mask]

                # 计算簇内所有点到中心的距离
                distances = np.linalg.norm(cluster_points - cluster_center, axis=1)

                # 使用百分位数作为边界阈值
                boundary_threshold = np.percentile(distances, boundary_percentile)
                self.cluster_boundary_thresholds[cluster_id] = boundary_threshold

                logger.info(f"簇 {cluster_id}: 恶意比例={malware_ratio:.2%}, "
                            f"样本数={len(cluster_true_labels)}, "
                            f"边界阈值={boundary_threshold:.4f}")
            else:
                cluster_malware_ratios[cluster_id] = 0
                self.cluster_boundary_thresholds[cluster_id] = float('inf')

        # 确定善意簇和恶意簇（基于恶意百分比）
        sorted_clusters = sorted(cluster_malware_ratios.items(), key=lambda x: x[1])

        # 善意簇：恶意比例最低的簇
        self.benign_cluster_id = sorted_clusters[0][0]
        # 恶意簇：恶意比例最高的簇
        self.malicious_cluster_id = sorted_clusters[-1][0]

        # 记录每个簇的类型
        self.cluster_types = {}
        self.cluster_types[self.benign_cluster_id] = "benign"
        self.cluster_types[self.malicious_cluster_id] = "malicious"

        # 如果有更多簇，中间簇标记为混合
        for cluster_id, ratio in sorted_clusters[1:-1]:
            self.cluster_types[cluster_id] = "mixed"

        logger.info(f"善意簇 ID: {self.benign_cluster_id} (恶意比例: {cluster_malware_ratios[self.benign_cluster_id]:.2%})")
        logger.info(f"恶意簇 ID: {self.malicious_cluster_id} (恶意比例: {cluster_malware_ratios[self.malicious_cluster_id]:.2%})")

        # 为边界距离计算初始化最近邻模型
        self.nn_model = NearestNeighbors(n_neighbors=min(5, len(test_embeddings)), metric='euclidean')
        self.nn_model.fit(test_embeddings)

    def compute_boundary_score(self, embedding: np.ndarray) -> Tuple[float, float, str]:
        """
        基于到善意和恶意簇边界的相对距离计算恶意性分数

        Returns:
            distance_to_boundary: 到所属簇边界的距离（正值表示在簇内）
            normalized_score: 归一化后的分数（0-1，越高越恶意）
            pred_class: 预测类别
        """
        # 计算到所有簇中心的距离
        distances_to_centers = np.linalg.norm(self.kmeans.cluster_centers_ - embedding, axis=1)

        # 找到最近的簇
        nearest_cluster = np.argmin(distances_to_centers)
        nearest_cluster_distance = distances_to_centers[nearest_cluster]

        # 获取该簇的边界阈值
        nearest_cluster_threshold = self.cluster_boundary_thresholds.get(nearest_cluster, float('inf'))

        # 计算到最近簇边界的距离（正值表示在簇内，负值表示在簇外）
        distance_to_boundary = nearest_cluster_threshold - nearest_cluster_distance

        # 如果点在簇外，设为0（视为边界点）
        if distance_to_boundary < 0:
            distance_to_boundary = 0

        # 计算到善意簇和恶意簇中心的距离
        dist_to_benign_center = distances_to_centers[self.benign_cluster_id]
        dist_to_malicious_center = distances_to_centers[self.malicious_cluster_id]

        # 获取善意簇和恶意簇的边界阈值
        benign_threshold = self.cluster_boundary_thresholds.get(self.benign_cluster_id, float('inf'))
        malicious_threshold = self.cluster_boundary_thresholds.get(self.malicious_cluster_id, float('inf'))

        # 计算到善意簇边界的距离（如果在内部则为正）
        dist_to_benign_boundary = benign_threshold - dist_to_benign_center
        dist_to_benign_boundary = max(0, dist_to_benign_boundary)

        # 计算到恶意簇边界的距离（如果在内部则为正）
        dist_to_malicious_boundary = malicious_threshold - dist_to_malicious_center
        dist_to_malicious_boundary = max(0, dist_to_malicious_boundary)

        # 计算到簇中心的相对距离（用于内部深度）
        benign_depth = dist_to_benign_boundary / (benign_threshold + 1e-8)
        malicious_depth = dist_to_malicious_boundary / (malicious_threshold + 1e-8)

        # 计算相对距离
        if dist_to_malicious_boundary > 0 or dist_to_benign_boundary > 0:
            # 如果至少在某个簇内
            if dist_to_malicious_boundary > 0 and dist_to_benign_boundary == 0:
                # 只在恶意簇内
                normalized_score = 0.5 + 0.5 * malicious_depth
            elif dist_to_benign_boundary > 0 and dist_to_malicious_boundary == 0:
                # 只在善意簇内
                normalized_score = 0.5 - 0.5 * benign_depth
            else:
                # 同时在两个簇内或都不在
                if dist_to_malicious_center < dist_to_benign_center:
                    # 更靠近恶意簇中心
                    normalized_score = 0.5 + 0.5 * malicious_depth
                else:
                    # 更靠近善意簇中心
                    normalized_score = 0.5 - 0.5 * benign_depth
        else:
            # 都不在簇内，基于到簇中心的距离
            total_dist = dist_to_benign_center + dist_to_malicious_center + 1e-8
            normalized_score = dist_to_benign_center / total_dist

        # 根据相对距离确定类别
        if dist_to_malicious_boundary > dist_to_benign_boundary:
            pred_class = "恶意"
        elif dist_to_benign_boundary > dist_to_malicious_boundary:
            pred_class = "良性"
        else:
            # 边界相等或都在外部，基于到中心的距离
            if dist_to_malicious_center < dist_to_benign_center:
                pred_class = "恶意"
            else:
                pred_class = "良性"

        # 确保分数在0-1范围内
        normalized_score = float(np.clip(normalized_score, 0, 1))

        return distance_to_boundary, normalized_score, pred_class

    def get_cluster_info(self) -> Dict[str, Any]:
        """获取聚类信息"""
        return {
            'n_clusters': self.n_clusters,
            'benign_cluster_id': self.benign_cluster_id,
            'malicious_cluster_id': self.malicious_cluster_id,
            'cluster_types': self.cluster_types,
            'cluster_boundary_thresholds': self.cluster_boundary_thresholds,
            'cluster_centers': self.kmeans.cluster_centers_
        }


# ==================== 模型加载函数 ====================
def load_contrastive_model(model_path: str, input_dim: int) -> Optional[nn.Module]:
    """加载训练好的对比学习模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(model_path):
        logger.warning(f"模型文件不存在: {model_path}，将使用原始特征")
        return None

    original_model = OriginalContrastiveModel(
        input_dim=input_dim,
        hidden_dim=Config.HIDDEN_DIM,
        output_dim=Config.OUTPUT_DIM
    )

    state_dict = torch.load(model_path, map_location='cpu')
    original_model.load_state_dict(state_dict)
    logger.info("原始模型权重加载成功")

    inference_model = ContrastiveModelInference(
        input_dim=input_dim,
        hidden_dim=Config.HIDDEN_DIM,
        output_dim=Config.OUTPUT_DIM
    )

    with torch.no_grad():
        inference_model.encoder[0].weight.copy_(original_model.encoder[0].weight)
        inference_model.encoder[0].bias.copy_(original_model.encoder[0].bias)
        inference_model.encoder[4].weight.copy_(original_model.encoder[4].weight)
        inference_model.encoder[4].bias.copy_(original_model.encoder[4].bias)
        inference_model.encoder[8].weight.copy_(original_model.encoder[8].weight)
        inference_model.encoder[8].bias.copy_(original_model.encoder[8].bias)

    inference_model.eval()
    inference_model.to(device)
    logger.info("推理模型构建成功")

    return inference_model


# ==================== 单样本推理函数 ====================
def encode_single_with_model(model: nn.Module, feature: np.ndarray) -> np.ndarray:
    """安全地使用模型编码单个样本"""
    device = next(model.parameters()).device
    feature_tensor = torch.FloatTensor(feature).unsqueeze(0).to(device)

    with torch.no_grad():
        if any(isinstance(m, nn.BatchNorm1d) for m in model.modules()):
            batch = feature_tensor.repeat(3, 1)
            output_batch = model(batch)
            embedding = output_batch[0].cpu().numpy()
        else:
            embedding = model(feature_tensor).cpu().numpy().squeeze()

    return embedding


# ==================== 多函数可视化 ====================
def visualize_multiple_functions(
        test_embeddings: np.ndarray,
        test_labels: List[int],
        new_embeddings: List[np.ndarray],
        function_infos: List[Dict],
        cluster_info: Dict[str, Any],
        save_path: str,
        method: str = 'tsne'
):
    """可视化多个新函数在测试集中的位置"""

    combined = np.vstack([test_embeddings] + new_embeddings)

    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(test_embeddings)-1))
        reduced = reducer.fit_transform(combined)
        title_suffix = "t-SNE"
    else:
        reducer = PCA(n_components=2, random_state=42)
        reduced = reducer.fit_transform(combined)
        var_ratio = reducer.explained_variance_ratio_.sum()
        title_suffix = f"PCA (var:{var_ratio:.2f})"

    test_reduced = reduced[:len(test_embeddings)]
    new_reduced = reduced[len(test_embeddings):]

    plt.figure(figsize=(16, 12))

    # 绘制测试集
    benign_idx = [i for i, l in enumerate(test_labels) if l == 0]
    malicious_idx = [i for i, l in enumerate(test_labels) if l == 1]

    if benign_idx:
        plt.scatter(
            test_reduced[benign_idx, 0], test_reduced[benign_idx, 1],
            c='blue', marker='o', label=f'Benign({len(benign_idx)})',
            alpha=0.5, s=40, edgecolors='white', linewidth=0.5
        )

    if malicious_idx:
        plt.scatter(
            test_reduced[malicious_idx, 0], test_reduced[malicious_idx, 1],
            c='red', marker='^', label=f'Malware ({len(malicious_idx)})',
            alpha=0.5, s=40, edgecolors='white', linewidth=0.5
        )

    # 绘制新函数
    colors = {'良性': 'green', '恶意': 'darkred'}
    markers = {'良性': 'D', '恶意': 'D'}

    for i, (info, coord) in enumerate(zip(function_infos, new_reduced)):
        pred_class = info['pred_class']
        plt.scatter(
            coord[0], coord[1],
            c=colors[pred_class], marker=markers[pred_class], s=200,
            label=f"{info['name']} ({pred_class}, {info['normalized_score']:.3f})",
            edgecolors='black', linewidth=2, zorder=10
        )

        # 添加标签
        plt.annotate(
            info['name'][:10],
            (coord[0], coord[1]),
            xytext=(5, 5), textcoords='offset points',
            fontsize=8, bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7)
        )

    plt.title(f'Multiple Functions Analysis - Relative Distance ({title_suffix})', fontsize=14)
    plt.legend(fontsize=9, loc='best', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"多函数可视化保存: {save_path}")


# ==================== 主分析函数 ====================
def analyze_functions(
        codes: List[str],
        function_names: List[str],
        config: Config = None
) -> List[Dict]:
    """分析多个函数"""

    if config is None:
        config = Config()

    if isinstance(codes, str):
        codes = [codes]
    if isinstance(function_names, str):
        function_names = [function_names]

    assert len(codes) == len(function_names), "代码列表和名称列表长度必须一致"

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # 1. 加载模型
    logger.info("="*50)
    logger.info("步骤1: 加载模型")

    deepseek_encoder = DeepSeekEncoder(config.MODEL_PATH)
    contrastive_model = load_contrastive_model(config.CONTRASTIVE_MODEL_PATH, config.INPUT_DIM)

    # 2. 加载测试集嵌入
    logger.info("="*50)
    logger.info("步骤2: 加载测试集数据")

    if not os.path.exists(config.TEST_EMBEDDINGS_PATH):
        logger.error(f"测试集嵌入文件不存在: {config.TEST_EMBEDDINGS_PATH}")
        return []

    test_embeddings = np.load(config.TEST_EMBEDDINGS_PATH)
    logger.info(f"测试集嵌入加载成功: {test_embeddings.shape}")

    if os.path.exists(config.TEST_LABELS_PATH):
        test_labels = np.load(config.TEST_LABELS_PATH).tolist()
        logger.info(f"测试集标签加载成功: {len(test_labels)}个")
    else:
        logger.warning(f"测试集标签文件不存在，使用默认标签")
        half = len(test_embeddings) // 2
        test_labels = [0] * half + [1] * (len(test_embeddings) - half)

    # 3. 初始化边界距离分析器
    logger.info("="*50)
    logger.info("步骤3: 初始化边界距离分析器")

    analyzer = BoundaryDistanceAnalyzer(
        test_embeddings,
        test_labels,
        n_clusters=config.N_CLUSTERS,
        boundary_percentile=config.BOUNDARY_PERCENTILE
    )

    cluster_info = analyzer.get_cluster_info()
    logger.info(f"聚类信息: 善意簇={cluster_info['benign_cluster_id']}, 恶意簇={cluster_info['malicious_cluster_id']}")

    # 4. 处理每个函数
    logger.info("="*50)
    logger.info(f"步骤4: 处理 {len(codes)} 个函数")

    all_embeddings = []
    all_results = []

    for idx, (code, name) in enumerate(zip(codes, function_names)):
        logger.info(f"\n处理函数 {idx+1}/{len(codes)}: {name}")

        # 向量化
        raw_feature = deepseek_encoder.encode(code)

        # 对比学习嵌入
        if contrastive_model:
            embedding = encode_single_with_model(contrastive_model, raw_feature)
        else:
            embedding = raw_feature

        # 边界距离分析
        distance_to_boundary, normalized_score, pred_class = analyzer.compute_boundary_score(embedding)

        logger.info(f"  边界内距离: {distance_to_boundary:.4f}")
        logger.info(f"  归一化分数: {normalized_score:.4f}, 预测: {pred_class}")

        result = {
            'name': name,
            'distance_to_boundary': float(distance_to_boundary),
            'normalized_score': float(normalized_score),
            'pred_class': pred_class,
            'code_preview': code[:100] + "..." if len(code) > 100 else code
        }

        all_embeddings.append(embedding)
        all_results.append(result)

        # 保存单个结果
        safe_name = name.replace('/', '_').replace('\\', '_')
        with open(f"{config.RESULTS_DIR}/{safe_name}_results.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # 5. 多函数可视化
    logger.info("="*50)
    logger.info("步骤5: 多函数可视化")

    visualize_multiple_functions(
        test_embeddings, test_labels,
        all_embeddings, all_results,
        cluster_info,
        save_path=f"{config.RESULTS_DIR}/multiple_functions_tsne.png",
        method='tsne'
    )

    # 同时保存PCA可视化作为备选
    visualize_multiple_functions(
        test_embeddings, test_labels,
        all_embeddings, all_results,
        cluster_info,
        save_path=f"{config.RESULTS_DIR}/multiple_functions_pca.png",
        method='pca'
    )

    # 6. 保存汇总结果
    summary = {
        'total_functions': len(all_results),
        'predictions': {
            'benign': sum(1 for r in all_results if r['pred_class'] == '良性'),
            'malicious': sum(1 for r in all_results if r['pred_class'] == '恶意'),
        },
        'cluster_info': {
            'benign_cluster_id': int(cluster_info['benign_cluster_id']),
            'malicious_cluster_id': int(cluster_info['malicious_cluster_id']),
            'cluster_types': {str(k): v for k, v in cluster_info['cluster_types'].items()},
            'cluster_boundary_thresholds': {str(k): float(v) for k, v in cluster_info['cluster_boundary_thresholds'].items()}
        },
        'details': all_results
    }

    with open(f"{config.RESULTS_DIR}/analysis_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("="*50)
    logger.info(f"分析完成! 结果汇总:")
    logger.info(f"  良性: {summary['predictions']['benign']} 个")
    logger.info(f"  恶意: {summary['predictions']['malicious']} 个")
    logger.info(f"汇总文件: {config.RESULTS_DIR}/analysis_summary.json")
    logger.info("="*50)

    return all_results


def main():
    parser = argparse.ArgumentParser(description='分析多个函数的恶意性（基于相对距离）')
    parser.add_argument('--files', nargs='+', help='包含函数代码的文件路径列表')
    parser.add_argument('--names', nargs='+', default=None, help='函数名称列表')
    parser.add_argument('--clusters', type=int, default=2, help='聚类数量（默认2）')
    parser.add_argument('--percentile', type=float, default=90, help='边界阈值百分位数（默认90）')

    args = parser.parse_args()

    # 示例函数
    codes = ['''.method public getView(ILandroid/view/View;Landroid/view/ViewGroup;)Landroid/view/View;
        ...''',  # 第一个函数（良性）
             '''.method public static a(Ljava/io/InputStream;Ljava/io/File;)V
             ...''']  # 第二个函数（恶意）

    codes_label = ['Benign_Function', 'Malware_Function']

    # 如果有文件参数，从文件读取
    if args.files:
        codes = []
        for file_path in args.files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    codes.append(f.read())
            except Exception as e:
                logger.error(f"读取文件失败 {file_path}: {e}")
                return

        if args.names and len(args.names) == len(codes):
            codes_label = args.names
        else:
            codes_label = [f"func_{i}" for i in range(len(codes))]

    # 更新配置
    config = Config()
    config.N_CLUSTERS = args.clusters
    config.BOUNDARY_PERCENTILE = args.percentile

    # 分析函数
    results = analyze_functions(codes, codes_label, config)

    # 打印简要结果
    if results:
        print("\n" + "="*70)
        print("分析结果汇总（基于相对距离）:")
        print("="*70)
        print(f"{'函数名称':20} | {'预测':4} | {'归一化分数':10} | {'边界内距离':12}")
        print("-"*70)
        for r in results:
            print(f"{r['name'][:20]:20} | {r['pred_class']:4} | {r['normalized_score']:.4f}     | {r['distance_to_boundary']:.4f}")
        print("="*70)
        print("注：边界内距离为正值（越内陆，分数越高）")
        print("归一化分数越高表示越恶意（0-1）")


if __name__ == "__main__":
    main()