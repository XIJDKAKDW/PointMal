#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
实验：基于对比学习的恶意代码函数嵌入优化
- 获取原始函数（善意/恶意）
- 10%测试集，90%训练集
- DeepSeek向量化
- 降维观察分布
- 对比学习优化嵌入（去除模糊恶意样本）
- 可视化对比学习前后测试集分布
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
import os
import logging
from collections import defaultdict
from typing import List, Dict, Tuple, Any
import random

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== 配置 ====================
class Config:
    MODEL_PATH = r"/home/changxiaosong/python/malwareTest/deepseek-coder-1.3b-base"
    JSON_FILE = r"./llm_features_tr.json"

    # 对比学习参数
    CONTRASTIVE_TEMPERATURE = 0.1
    CONTRASTIVE_MARGIN = 1.0
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 20
    HIDDEN_DIM = 256
    OUTPUT_DIM = 128

    # 模糊样本阈值（欧氏距离）
    AMBIGUOUS_THRESHOLD = 1.5

    # 随机种子
    RANDOM_SEED = 42


# ==================== DeepSeek编码器 ====================
class DeepSeekEncoder:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"加载模型到 {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32
        ).to(self.device)
        self.model.eval()
        logger.info("模型加载完成")

    def encode(self, code_snippets: List[str], max_length: int = 256) -> np.ndarray:
        """批量获取代码向量（平均池化）"""
        embeddings = []

        for code in code_snippets:
            inputs = self.tokenizer(
                code,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]  # 最后一层
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                embedding = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
                embeddings.append(embedding.cpu().numpy())

        return np.vstack(embeddings)


# ==================== 数据加载 ====================
def load_methods_from_json(json_file: str) -> Tuple[List[str], List[int], List[str]]:
    """从JSON加载函数代码和标签"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    methods = []
    labels = []
    sample_ids = []

    def split_methods(rm_str: str) -> List[str]:
        if not rm_str:
            return []
        methods = rm_str.split('@@@cxs@@@')
        return [m.strip() for m in methods if m.strip() and len(m) > 20][:50]

    for seq, sample in data.items():
        label = sample.get('true_label')
        if label not in (0, 1):
            continue

        method_list = split_methods(sample.get('RM_str', ''))
        for method in method_list:
            methods.append(method)
            labels.append(label)
            sample_ids.append(seq)

    logger.info(f"加载 {len(methods)} 个函数 (良性:{labels.count(0)}, 恶意:{labels.count(1)})")
    return methods, labels, sample_ids


# ==================== 对比学习数据集 ====================
class ContrastiveDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: List[int], distances: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.distances = distances

        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        anchor_feat = self.features[idx]
        anchor_label = self.labels[idx].item()

        # 选择正样本（同标签）
        same_label_indices = self.label_to_indices[anchor_label]
        if len(same_label_indices) > 1:
            positive_idx = np.random.choice([i for i in same_label_indices if i != idx])
        else:
            # 如果没有其他同标签样本，随机选一个不同的
            positive_idx = np.random.choice([i for i in range(len(self.features)) if i != idx])

        # 选择负样本（不同标签）
        other_labels = [l for l in self.label_to_indices.keys() if l != anchor_label]
        if other_labels:
            negative_label = np.random.choice(other_labels)
            negative_idx = np.random.choice(self.label_to_indices[negative_label])
        else:
            negative_idx = np.random.choice([i for i in range(len(self.features)) if i != idx])

        return anchor_feat, self.features[positive_idx], self.features[negative_idx]


# ==================== 对比学习模型 ====================
class ContrastiveModel(nn.Module):
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


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        losses = F.relu(pos_dist - neg_dist + self.margin)
        return losses.mean()


class ContrastiveTrainer:
    def __init__(self, model: nn.Module, config: Config):
        self.model = model
        self.config = config
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.criterion = TripletLoss(margin=config.CONTRASTIVE_MARGIN)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            anchor, positive, negative = [x.to(self.device) for x in batch]

            self.optimizer.zero_grad()

            anchor_out = self.model(anchor)
            positive_out = self.model(positive)
            negative_out = self.model(negative)

            loss = self.criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(self, train_dataset: ContrastiveDataset) -> List[float]:
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True
        )

        history = []
        for epoch in range(self.config.EPOCHS):
            loss = self.train_epoch(train_loader)
            history.append(loss)
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{self.config.EPOCHS} - Loss: {loss:.4f}")

        return history

    def encode(self, features: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(features).to(self.device)
            return self.model(tensor).cpu().numpy()


# ==================== 模糊样本识别 ====================
def identify_ambiguous_samples(
        features: np.ndarray,
        labels: List[int],
        threshold: float
) -> List[int]:
    """识别不同标签但特征相似的模糊样本，返回要移除的恶意样本索引"""
    distances = euclidean_distances(features)
    n = len(labels)

    ambiguous_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] != labels[j] and distances[i, j] < threshold:
                ambiguous_pairs.append((i, j, distances[i, j]))

    logger.info(f"发现 {len(ambiguous_pairs)} 个模糊样本对")

    # 移除恶意样本
    to_remove = set()
    for i, j, _ in ambiguous_pairs:
        if labels[i] == 1:
            to_remove.add(i)
        elif labels[j] == 1:
            to_remove.add(j)

    logger.info(f"将移除 {len(to_remove)} 个模糊恶意样本")
    return list(to_remove)

# ==================== Visualization ====================
def visualize_embeddings(
        embeddings: np.ndarray,
        labels: List[int],
        title: str,
        save_path: str,
        method: str = 'pca'
):
    """Visualize embedding distribution"""
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        reduced = reducer.fit_transform(embeddings)
        explained_var = reducer.explained_variance_ratio_.sum()
        title += f" (PCA, var:{explained_var:.2f})"
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced = reducer.fit_transform(embeddings)
        title += " (t-SNE)"

    plt.figure(figsize=(12, 8))

    benign_idx = [i for i, l in enumerate(labels) if l == 0]
    malicious_idx = [i for i, l in enumerate(labels) if l == 1]

    if benign_idx:
        plt.scatter(
            reduced[benign_idx, 0], reduced[benign_idx, 1],
            c='blue', marker='o', label=f'Benign ({len(benign_idx)})', alpha=0.6, s=30
        )

    if malicious_idx:
        plt.scatter(
            reduced[malicious_idx, 0], reduced[malicious_idx, 1],
            c='red', marker='^', label=f'Malicious ({len(malicious_idx)})', alpha=0.6, s=30
        )

    plt.title(title, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Visualization saved: {save_path}")

# ==================== 主实验 ====================
def main():
    config = Config()
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    # 1. 加载数据
    logger.info("="*50)
    logger.info("步骤1: 加载数据")
    methods, labels, sample_ids = load_methods_from_json(config.JSON_FILE)

    # 2. 划分训练集和测试集
    logger.info("="*50)
    logger.info("步骤2: 划分数据集 (90%训练, 10%测试)")
    train_methods, test_methods, train_labels, test_labels, train_ids, test_ids = train_test_split(
        methods, labels, sample_ids, test_size=0.1, random_state=config.RANDOM_SEED, stratify=labels
    )
    logger.info(f"训练集: {len(train_methods)} (良性:{train_labels.count(0)}, 恶意:{train_labels.count(1)})")
    logger.info(f"测试集: {len(test_methods)} (良性:{test_labels.count(0)}, 恶意:{test_labels.count(1)})")

    # 3. DeepSeek向量化
    logger.info("="*50)
    logger.info("步骤3: DeepSeek向量化")
    encoder = DeepSeekEncoder(config.MODEL_PATH)

    logger.info("编码训练集...")
    train_features = encoder.encode(train_methods)
    logger.info(f"训练集特征维度: {train_features.shape}")

    logger.info("编码测试集...")
    test_features = encoder.encode(test_methods)
    logger.info(f"测试集特征维度: {test_features.shape}")

    # 4. 降维观察原始分布
    logger.info("="*50)
    logger.info("步骤4: 原始嵌入分布可视化")
    os.makedirs("visualizations", exist_ok=True)

    # 训练集原始分布
    visualize_embeddings(
        train_features, train_labels,
        title="训练集原始嵌入分布 (DeepSeek)",
        save_path="visualizations/train_original_pca.png",
        method='pca'
    )
    visualize_embeddings(
        train_features, train_labels,
        title="训练集原始嵌入分布 (DeepSeek)",
        save_path="visualizations/train_original_tsne.png",
        method='tsne'
    )

    # 测试集原始分布
    visualize_embeddings(
        test_features, test_labels,
        title="测试集原始嵌入分布 (DeepSeek)",
        save_path="visualizations/test_original_pca.png",
        method='pca'
    )

    # 5. 识别模糊样本并清理训练集
    logger.info("="*50)
    logger.info("步骤5: 识别模糊样本并清理训练集")
    to_remove = identify_ambiguous_samples(
        train_features, train_labels,
        threshold=config.AMBIGUOUS_THRESHOLD
    )

    # 创建清理后的训练集
    clean_indices = [i for i in range(len(train_features)) if i not in to_remove]
    clean_train_features = train_features[clean_indices]
    clean_train_labels = [train_labels[i] for i in clean_indices]

    logger.info(f"清理后训练集: {len(clean_train_features)} (良性:{clean_train_labels.count(0)}, 恶意:{clean_train_labels.count(1)})")

    # 计算距离矩阵用于对比学习
    logger.info("计算距离矩阵...")
    from scipy.spatial.distance import pdist, squareform
    train_distances = squareform(pdist(clean_train_features, metric='euclidean'))

    # 6. 对比学习训练
    logger.info("="*50)
    logger.info("步骤6: 对比学习训练")

    train_dataset = ContrastiveDataset(clean_train_features, clean_train_labels, train_distances)

    contrastive_model = ContrastiveModel(
        input_dim=clean_train_features.shape[1],
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM
    )

    trainer = ContrastiveTrainer(contrastive_model, config)
    history = trainer.train(train_dataset)

    # 绘制训练损失
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(history)+1), history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('对比学习训练损失')
    plt.grid(True, alpha=0.3)
    plt.savefig('visualizations/training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 7. 获取对比学习后的嵌入
    logger.info("="*50)
    logger.info("步骤7: 获取对比学习后的嵌入")

    # 训练集嵌入
    train_contrastive = trainer.encode(clean_train_features)

    # 测试集嵌入（使用训练好的模型）
    test_contrastive = trainer.encode(test_features)

    # 8. 可视化对比学习后的测试集分布
    logger.info("="*50)
    logger.info("步骤8: 对比学习后测试集分布可视化")

    # 测试集原始 vs 对比学习后（并排对比）
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # 原始分布
    pca_original = PCA(n_components=2, random_state=42)
    test_original_pca = pca_original.fit_transform(test_features)

    benign_idx = [i for i, l in enumerate(test_labels) if l == 0]
    malicious_idx = [i for i, l in enumerate(test_labels) if l == 1]

    axes[0].scatter(
        test_original_pca[benign_idx, 0], test_original_pca[benign_idx, 1],
        c='blue', marker='o', label=f'良性 ({len(benign_idx)})', alpha=0.6, s=30
    )
    axes[0].scatter(
        test_original_pca[malicious_idx, 0], test_original_pca[malicious_idx, 1],
        c='red', marker='^', label=f'恶意 ({len(malicious_idx)})', alpha=0.6, s=30
    )
    axes[0].set_title('测试集原始嵌入 (PCA)', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 单独保存对比学习后的t-SNE
    visualize_embeddings(
        test_contrastive, test_labels,
        title="测试集对比学习后嵌入 (t-SNE)",
        save_path="visualizations/test_contrastive_tsne.png",
        method='tsne'
    )

    # 9. 计算类间距离对比
    logger.info("="*50)
    logger.info("步骤9: 类间距离对比")

    def compute_inter_class_distance(features, labels):
        benign_features = features[[i for i, l in enumerate(labels) if l == 0]]
        malicious_features = features[[i for i, l in enumerate(labels) if l == 1]]

        if len(benign_features) == 0 or len(malicious_features) == 0:
            return 0

        # 计算类中心
        benign_center = benign_features.mean(axis=0)
        malicious_center = malicious_features.mean(axis=0)

        # 欧氏距离
        return np.linalg.norm(benign_center - malicious_center)

    original_inter_dist = compute_inter_class_distance(test_features, test_labels)
    contrastive_inter_dist = compute_inter_class_distance(test_contrastive, test_labels)

    logger.info(f"测试集类间距离 (原始): {original_inter_dist:.4f}")
    logger.info(f"测试集类间距离 (对比学习后): {contrastive_inter_dist:.4f}")
    logger.info(f"距离提升: {contrastive_inter_dist/original_inter_dist:.2f}倍")

    # 10. 保存结果
    logger.info("="*50)
    logger.info("步骤10: 保存结果")

    results = {
        'config': {
            'threshold': config.AMBIGUOUS_THRESHOLD,
            'epochs': config.EPOCHS,
            'output_dim': config.OUTPUT_DIM,
            'margin': config.CONTRASTIVE_MARGIN
        },
        'data_stats': {
            'total_methods': len(methods),
            'train_original': {'total': len(train_features), 'benign': train_labels.count(0), 'malicious': train_labels.count(1)},
            'train_cleaned': {'total': len(clean_train_features), 'benign': clean_train_labels.count(0), 'malicious': clean_train_labels.count(1)},
            'test': {'total': len(test_features), 'benign': test_labels.count(0), 'malicious': test_labels.count(1)},
            'removed_ambiguous': len(to_remove)
        },
        'inter_class_distances': {
            'original': float(original_inter_dist),
            'contrastive': float(contrastive_inter_dist),
            'improvement_ratio': float(contrastive_inter_dist/original_inter_dist)
        }
    }

    with open('visualizations/experiment_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # 保存模型
    torch.save(contrastive_model.state_dict(), 'visualizations/contrastive_model.pth')
    np.save('visualizations/test_contrastive_embeddings.npy', test_contrastive)

    logger.info("实验完成！所有结果保存在 visualizations/ 目录")


if __name__ == "__main__":
    main()