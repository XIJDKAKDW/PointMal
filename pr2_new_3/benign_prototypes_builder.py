#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：malwareTest 
@File    ：benign_prototypes_builder.py
@IDE     ：PyCharm 
@Author  ：常晓松
@Date    ：2026/2/13 11:33 
'''
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
良性函数特征原型构建器
用于从指定文件夹中的txt文件（每个文件一个Smali函数）构建特征原型集合

输入: 包含良性函数Smali代码的文件夹路径，每个函数一个txt文件
输出: 良性函数特征原型文件 (benign_prototypes.pkl)
"""

import argparse
import os
import pickle
import re
from typing import Dict, List

import numpy as np
from tqdm import tqdm

# 导入轻量级嵌入模型
from test001Method_new_9_4_3 import LightweightEmbeddingModel


class BenignPrototypeBuilder:
    """良性函数特征原型构建器"""

    def __init__(self,
                 embedding_model_name='glove-wiki-gigaword-100',
                 embedding_dim=100,
                 output_file='./benign_prototypes.pkl'):
        """
        初始化原型构建器

        Args:
            embedding_model_name: 预训练嵌入模型名称
            embedding_dim: 嵌入维度
            output_file: 输出原型文件路径
        """
        self.embedding_model = LightweightEmbeddingModel(
            model_name=embedding_model_name,
            embedding_dim=embedding_dim
        )
        self.output_file = output_file
        self.embedding_dim = self.embedding_model.embedding_dim

    def read_smali_file(self, file_path: str) -> str:
        """
        读取Smali文件并提取方法体

        Args:
            file_path: Smali文件路径

        Returns:
            提取的方法代码文本
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 提取方法体
            method_pattern = r'\.method.*?\n(.*?)\.end method'
            matches = re.findall(method_pattern, content, re.DOTALL)

            if matches:
                # 如果有多个方法，合并所有方法
                method_bodies = []
                for match in matches:
                    # 提取指令部分，去除注释和空行
                    lines = match.strip().split('\n')
                    instructions = []
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # 简化指令，移除寄存器
                            parts = line.split(' ', 2)
                            if len(parts) >= 2:
                                opcode = parts[1] if parts[0] in ['invoke-virtual', 'invoke-direct', 'invoke-static',
                                                                  'invoke-interface', 'invoke-super', 'const',
                                                                  'move', 'if', 'goto', 'return'] else parts[0]
                                instructions.append(opcode)
                    method_bodies.append(' '.join(instructions))

                return ' '.join(method_bodies)
            else:
                # 如果没有找到方法，返回整个文件内容
                return content

        except Exception as e:
            print(f"读取文件 {file_path} 失败: {e}")
            return ""

    def extract_function_name_from_file(self, file_path: str) -> str:
        """
        从文件路径提取函数名
        """
        filename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(filename)[0]

        # 尝试从文件名中提取有意义的名称
        # 格式可能为: class_method.smali 或 method.smali
        if '_' in name_without_ext:
            parts = name_without_ext.split('_')
            return parts[-1]  # 取最后一部分作为方法名
        else:
            return name_without_ext

    def build_prototypes_from_directory(self, benign_dir: str,
                                        use_clustering: bool = True,
                                        n_clusters: int = 50,
                                        similarity_threshold: float = 0.8) -> Dict:
        """
        从良性函数目录构建特征原型

        Args:
            benign_dir: 良性函数txt文件所在目录
            use_clustering: 是否使用聚类（否则使用平均池化）
            n_clusters: 聚类中心数量
            similarity_threshold: 相似度阈值（用于去重）

        Returns:
            良性原型字典 {prototype_id: embedding_vector}
        """
        print(f"开始从目录构建良性函数特征原型: {benign_dir}")

        # 获取所有txt文件
        txt_files = []
        for root, dirs, files in os.walk(benign_dir):
            for file in files:
                if file.endswith('.txt'):
                    txt_files.append(os.path.join(root, file))

        print(f"找到 {len(txt_files)} 个良性函数文件")

        if len(txt_files) == 0:
            print("警告: 没有找到任何txt文件")
            return {}

        # 提取所有函数的嵌入向量
        embeddings = []
        file_names = []

        for file_path in tqdm(txt_files, desc="提取函数嵌入"):
            # 读取Smali代码
            code_text = self.read_smali_file(file_path)

            if code_text.strip():
                # 获取嵌入向量
                embedding = self.embedding_model.embed_text(code_text, pooling='mean')
                embeddings.append(embedding)
                file_names.append(os.path.basename(file_path))

        if len(embeddings) == 0:
            print("错误: 没有成功提取任何嵌入向量")
            return {}

        embeddings = np.array(embeddings)
        print(f"成功提取 {len(embeddings)} 个嵌入向量，维度: {embeddings.shape[1]}")

        # 去重：移除相似度过高的向量
        if len(embeddings) > 1:
            print("正在进行相似度去重...")
            unique_embeddings = []
            indices_to_keep = []

            for i in range(len(embeddings)):
                is_duplicate = False
                for j in range(len(unique_embeddings)):
                    # 计算余弦相似度
                    emb_i = embeddings[i]
                    emb_j = unique_embeddings[j]

                    norm_i = np.linalg.norm(emb_i)
                    norm_j = np.linalg.norm(emb_j)

                    if norm_i > 0 and norm_j > 0:
                        similarity = np.dot(emb_i, emb_j) / (norm_i * norm_j)
                        if similarity > similarity_threshold:
                            is_duplicate = True
                            break

                if not is_duplicate:
                    unique_embeddings.append(embeddings[i])
                    indices_to_keep.append(i)

            embeddings = np.array(unique_embeddings)
            print(f"去重后保留 {len(embeddings)} 个独特嵌入向量")

        prototypes = {}

        if use_clustering and len(embeddings) >= n_clusters:
            # 使用K-Means聚类
            print(f"使用K-Means聚类 (n_clusters={n_clusters})...")
            try:
                from sklearn.cluster import KMeans

                # 调整聚类中心数量
                n_clusters = min(n_clusters, len(embeddings))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(embeddings)

                # 存储聚类中心作为原型
                for i, center in enumerate(kmeans.cluster_centers_):
                    prototypes[f"cluster_{i}"] = center

                print(f"聚类完成，生成 {len(prototypes)} 个原型")

            except ImportError:
                print("警告: sklearn未安装，使用随机采样代替聚类")
                use_clustering = False

        if not use_clustering:
            # 使用随机采样或所有向量
            if len(embeddings) > 100:
                # 随机采样100个作为原型
                indices = np.random.choice(len(embeddings), 100, replace=False)
                for i, idx in enumerate(indices):
                    prototypes[f"sample_{i}"] = embeddings[idx]
            else:
                # 全部作为原型
                for i, emb in enumerate(embeddings):
                    prototypes[f"prototype_{i}"] = emb

            print(f"生成 {len(prototypes)} 个原型（采样）")

        # 保存原型到文件
        self.save_prototypes(prototypes)

        return prototypes

    def build_prototypes_from_embeddings(self, embeddings: List[np.ndarray]) -> Dict:
        """
        直接从嵌入向量列表构建原型

        Args:
            embeddings: 嵌入向量列表

        Returns:
            良性原型字典
        """
        prototypes = {}

        if len(embeddings) > 100:
            # 随机采样100个
            indices = np.random.choice(len(embeddings), 100, replace=False)
            for i, idx in enumerate(indices):
                prototypes[f"prototype_{i}"] = embeddings[idx]
        else:
            for i, emb in enumerate(embeddings):
                prototypes[f"prototype_{i}"] = emb

        return prototypes

    def save_prototypes(self, prototypes: Dict):
        """
        保存原型到文件

        Args:
            prototypes: 原型字典
        """
        try:
            # 保存完整数据
            save_data = {
                'prototypes': prototypes,
                'embedding_dim': self.embedding_dim,
                'model_name': self.embedding_model.model_name,
                'count': len(prototypes)
            }

            with open(self.output_file, 'wb') as f:
                pickle.dump(save_data, f)

            print(f"成功保存 {len(prototypes)} 个良性原型到: {self.output_file}")

        except Exception as e:
            print(f"保存原型失败: {e}")

    def load_prototypes(self, prototype_file: str = None) -> Dict:
        """
        加载原型文件

        Args:
            prototype_file: 原型文件路径（默认使用初始化时的output_file）

        Returns:
            原型字典
        """
        load_file = prototype_file or self.output_file

        if os.path.exists(load_file):
            try:
                with open(load_file, 'rb') as f:
                    data = pickle.load(f)

                if isinstance(data, dict):
                    if 'prototypes' in data:
                        prototypes = data['prototypes']
                    else:
                        prototypes = data
                else:
                    prototypes = {}

                print(f"成功加载 {len(prototypes)} 个良性原型")
                return prototypes

            except Exception as e:
                print(f"加载原型失败: {e}")
                return {}
        else:
            print(f"原型文件不存在: {load_file}")
            return {}


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='构建良性函数特征原型')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='良性函数txt文件所在目录')
    parser.add_argument('--output', type=str, default='./benign_prototypes.pkl',
                        help='输出原型文件路径')
    parser.add_argument('--model', type=str, default='glove-wiki-gigaword-100',
                        help='预训练嵌入模型名称')
    parser.add_argument('--dim', type=int, default=100,
                        help='嵌入维度')
    parser.add_argument('--clusters', type=int, default=50,
                        help='聚类中心数量')
    parser.add_argument('--no_clustering', action='store_true',
                        help='不使用聚类，使用采样')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='相似度去重阈值')

    args = parser.parse_args()

    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        return

    # 创建原型构建器
    builder = BenignPrototypeBuilder(
        embedding_model_name=args.model,
        embedding_dim=args.dim,
        output_file=args.output
    )

    # 构建原型
    prototypes = builder.build_prototypes_from_directory(
        benign_dir=args.input_dir,
        use_clustering=not args.no_clustering,
        n_clusters=args.clusters,
        similarity_threshold=args.threshold
    )

    print(f"\n原型构建完成!")
    print(f"  - 原型数量: {len(prototypes)}")
    print(f"  - 嵌入维度: {builder.embedding_dim}")
    print(f"  - 输出文件: {args.output}")
    print(f"  - 嵌入模型: {args.model}")

    # 显示原型统计信息
    if prototypes:
        # 计算原型间的平均相似度
        proto_list = list(prototypes.values())
        if len(proto_list) > 1:
            similarities = []
            for i in range(min(10, len(proto_list))):
                for j in range(i+1, min(10, len(proto_list))):
                    emb_i = proto_list[i]
                    emb_j = proto_list[j]
                    norm_i = np.linalg.norm(emb_i)
                    norm_j = np.linalg.norm(emb_j)
                    if norm_i > 0 and norm_j > 0:
                        sim = np.dot(emb_i, emb_j) / (norm_i * norm_j)
                        similarities.append(sim)

            if similarities:
                print(f"  - 原型间平均相似度: {np.mean(similarities):.4f}")


if __name__ == "__main__":
    main()