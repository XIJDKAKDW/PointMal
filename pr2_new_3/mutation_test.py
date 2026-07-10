#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：malwareTest 
@File    ：mutation_test.py
@IDE     ：PyCharm 
@Author  ：常晓松
@Date    ：2026/1/14 14:43 
'''
import json
import logging
# coding=utf-8
import os
import platform
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
system = platform.system()

if system == "Linux":
    sys.path.append(r"/home/changxiaosong/python/malwareTest")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr2_new_3")

from pr2_new_3.test001Method_new_9_4_3 import llm_chat
from pr2_new_3.test004 import get_label_loop


# 设置日志
from pr2_new_3.test001 import get_connection
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
@dataclass
class RMData:
    """存储RM数据"""
    seq: str
    content: str
    true_label: int  # 0: benign, 1: malicious
    predicted_label: int  # LLM预测标签
    confidence: float  # 置信度
    file_path: str  # RM文件路径

@dataclass
class OperationSet:
    """操作集合"""
    operations: ''  # 1-17的操作码集合
    fitness: float = 0.0  # 适应度(F1分数)
    age: int = 0  # 存活代数
    usage_count: int = 0  # 使用次数
    def __init__(self,operations,fitness=0.0,age=0,usage_count=0):
        self.operations=operations
        self.fitness=fitness
        self.age=age
        self.usage_count=usage_count
    def __str__(self):
        ops = self.operations
        return f"Ops:{ops}, F1:{self.fitness:.4f}, Age:{self.age}, Used:{self.usage_count}"

class JavaCodeTransformer:
    def __init__(self, test002_path: str = "Test002",
                 javaparser_path: str = r"/home/changxiaosong/python/malwareTest/pr2_new_2/javaparser-core-3.25.9.jar"):
        self.test002_path = test002_path
        self.javaparser_path = javaparser_path
        self.system = platform.system()

    def transform_code(self, operations: Set[int], input_file: str, output_file: str = None) -> str:
        try:
            # 构造命令 - 传入二进制字符串
            if self.system == "Windows":
                classpath = f".;{self.javaparser_path}"
                command = f"java -cp {classpath} testPkg.Test002 {operations} {input_file}"
            else:
                classpath = f".:{self.javaparser_path}"
                command = f"java -cp {classpath} testPkg.Test002 {operations} {input_file}"


            # 执行命令
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            # logger.info(f"执行命令: {command}"+str(result))

            if result.returncode != 0:
                logger.error(f"Test002执行失败: {result.stderr}" + command)
                exit(0)
                return None

            # Test002会自动生成_ret文件
            if not output_file:
                base, ext = os.path.splitext(input_file)
                output_file = f"{base}_ret{ext}"

            if os.path.exists(output_file):
                return output_file

        except Exception as e:
            logger.error(f"代码转换失败: {e}")
            return None

class LLMAdapter:
    """LLM适配器，用于获取预测标签 """
    #"gemma3:1b"
    def __init__(self, llm_name: str = 'deepseek-coder-v2:16b'):
        self.llm_name = llm_name

    def predict_label(self, seq,transformed_code) -> Tuple[int, float]:
        """
        调用LLM获取预测标签

        Args:
            code_content: 代码内容

        Returns:
            (预测标签, 置信度)
        """
        task = f"""
As a professional malware analysis expert, please make a judgment based on the codes.
critical codes: [{transformed_code}]
## Output Format (Strictly follow this format):
Final Classification: [Benign/Malicious]
Key Evidence: [List the 2-3 most important features and their maliciousness indications]
Behavior Pattern: [Identified malicious behavior patterns, such as mining, ransomware, etc.]
Detailed Reasoning: [Complete analysis reasoning process, explaining how conclusions are drawn from features]
Final Confidence: [Comprehensive confidence based on all evidence: High/Medium/Low]
"""
        file_path= 'example5.txt'
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        lsi=content.split('\n\n')
        lsi_new=[]
        lsi_new.extend(lsi[0].split('Output:\n'))
        lsi_new.extend(lsi[1].split('Output:\n'))
        talks_2=[]
        talks_2.append( {"role": "user", "content": lsi_new[0]})
        talks_2.append({"role": "assistant", "content": lsi_new[1]})
        talks_2.append( {"role": "user", "content": lsi_new[2]})
        talks_2.append({"role": "assistant", "content": lsi_new[3]})
        #llm存在一定的随机性，若原样输出，则重新调用
        for i in range(5):
            talks_2, llm_ret = llm_chat(seq, talks_2.copy(), task, self.llm_name)
            if 'Final Classification: [Benign/Malicious]' not in llm_ret:
                break
        label = get_label_loop(llm_ret,'','')
        confidence=self.get_confidence(llm_ret)
        return label, confidence
        # return 1 ,"High"
    def get_confidence(self,llm_ret):
        label=''
        for one in llm_ret.split('\n'):
            if 'Final Confidence' in one:
                label=one
                break
        return 'High' if 'High' in label else 'Low'



class GeneticOptimizer:
    """遗传算法优化器"""

    def __init__(self,
                 population_size: int = 20,
                 max_generations: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.2,
                 max_age: int = 10,
                 stability_threshold: int = 10,
                 target_f1: float = 0.95,
                 eval_sample_size: int = 10,  # ← 新增：每次评估采样数量
                 eval_sample_ratio: float = 0.5,
                 best_eval_repeats: int = 3
                 ):

        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_age = max_age
        self.stability_threshold = stability_threshold
        self.target_f1 = target_f1
        self.eval_sample_size = eval_sample_size  # ← 新增
        self.eval_sample_ratio = eval_sample_ratio  # ← 新增

        self.population: List[OperationSet] = []
        self.best_solution: Optional[OperationSet] = None
        self.fitness_history: List[float] = []
        self.best_eval_repeats = best_eval_repeats
    def _sample_balanced_data(self, rm_samples: List[RMData]) -> List[RMData]:
        # 按标签分组
        malicious = [d for d in rm_samples if d.true_label == 1]
        benign = [d for d in rm_samples if d.true_label == 0]

        logger.debug(f"采样前: 恶意样本={len(malicious)}, 良性样本={len(benign)}")

        # 计算采样数量
        n_samples = min(self.eval_sample_size, len(rm_samples))

        # 根据比例计算正负样本数量
        n_mal = min(len(malicious), max(1, int(n_samples * self.eval_sample_ratio)))
        n_ben = min(len(benign), max(1, n_samples - n_mal))

        # 如果某一类样本不足，调整另一类样本数量
        if n_mal == 0:
            n_ben = min(len(benign), n_samples)
        elif n_ben == 0:
            n_mal = min(len(malicious), n_samples)

        # 随机采样
        sampled = []
        if malicious and n_mal > 0:
            sampled.extend(random.sample(malicious, n_mal))
        if benign and n_ben > 0:
            sampled.extend(random.sample(benign, n_ben))

        # 如果采样数量还不够，从剩余样本中补充
        if len(sampled) < n_samples:
            remaining = [d for d in rm_samples if d not in sampled]
            if remaining:
                needed = n_samples - len(sampled)
                sampled.extend(random.sample(remaining, min(needed, len(remaining))))

        # 打乱顺序
        random.shuffle(sampled)

        logger.info(f"采样后: 总计={len(sampled)}, 恶意={sum(1 for d in sampled if d.true_label==1)}, "
                     f"良性={sum(1 for d in sampled if d.true_label==0)}")

        return sampled
    def initialize_population(self, min_ops: int = 1, max_ops: int = 5):
        """初始化种群 - 使用二进制表示的操作集合"""
        self.population = []
        for _ in range(self.population_size-len(self.population)):
            # 创建17位二进制字符串
            binary_list = ['0'] * 17

            # 随机选择要设置为1的位数
            n_ops = random.randint(min_ops, max_ops)

            # 随机选择哪些位置为1
            positions = random.sample(range(17), n_ops)
            for pos in positions:
                binary_list[pos] = '1'

            # 将二进制字符串转换为操作集合
            binary_str = ''.join(binary_list)

            self.population.append(OperationSet(operations=binary_str))
        logger.info(f"初始化种群，大小: {self.population_size}")
    def evaluate_fitness(self, operation_set: OperationSet,
                         rm_samples: List[RMData],
                         transformer: JavaCodeTransformer,
                         llm_adapter: LLMAdapter,
                         is_best_candidate: bool = False) -> float:
        """
        评估操作集合的适应度(F1分数)

        Args:
            operation_set: 操作集合
            rm_samples: RM样本列表
            transformer: 代码转换器
            llm_adapter: LLM适配器

        Returns:
            F1分数
        """
        if not rm_samples:
            return 0.0
        n_repeats = 1#self.best_eval_repeats if is_best_candidate else 1
        sampled_data = self._sample_balanced_data(rm_samples)
        predictions = []
        true_labels = []
        print('cxs1',type(operation_set),operation_set)
        logger.info(f"评估操作集 {operation_set.operations}，使用 {len(sampled_data)} 个样本")
        for i, rm_data in enumerate(rm_samples):
            # 创建临时文件
            temp_input = rm_data.file_path
            with open(temp_input, 'w', encoding='utf-8') as f:
                f.write(rm_data.content)

            # 转换代码
            temp_output = transformer.transform_code(operation_set.operations, temp_input)
            if temp_output and os.path.exists(temp_output):
                # 读取转换后的代码
                with open(temp_output, 'r', encoding='utf-8') as f:
                    transformed_code = f.read()

                # LLM预测
                pred_label_tmp=[]
                confidence_tmp=[]
                for repeat in range(n_repeats):
                    pred_label, confidence = llm_adapter.predict_label(rm_data.seq,transformed_code)
                    pred_label_tmp.append(pred_label)
                    confidence_tmp.append(confidence)
                counter = Counter(pred_label_tmp)
                pred_label = counter.most_common(1)[0][0]
                counter = Counter(confidence_tmp)
                confidence = counter.most_common(1)[0][0]
                predictions.append(pred_label)
                true_labels.append(rm_data.true_label)

            # 清理临时文件
            # try:
            #     os.remove(temp_input)
            #     os.remove(temp_output)
            # except:
            #     pass

        # 计算F1分数
        if len(predictions) == 0:
            return 0.0

        f1_score = self._calculate_f1_score(true_labels, predictions)
        operation_set.fitness = f1_score
        operation_set.usage_count += 1

        return f1_score

    def _calculate_f1_score(self, true_labels: List[int], pred_labels: List[int]) -> float:
        """计算F1分数"""
        from sklearn.metrics import f1_score
        return f1_score(true_labels, pred_labels, zero_division=0)

    def selection(self) -> List[OperationSet]:
        """选择操作（轮盘赌选择）"""
        selected = []

        # 计算适应度和概率
        fitness_values = [op.fitness for op in self.population]
        fitness_sum = sum(fitness_values)

        # 如果所有适应度都为0，随机选择
        if fitness_sum == 0:
            return random.sample(self.population, self.population_size)

        # 计算概率
        probs = [fitness / fitness_sum for fitness in fitness_values]

        # 总是使用可重复选择，避免数值问题
        indices = np.random.choice(
            len(self.population),
            size=self.population_size,
            p=probs,
            replace=True  # 改为可重复选择
        )

        return [self.population[i] for i in indices]

    def crossover(self, parent1: OperationSet, parent2: OperationSet) -> Tuple[OperationSet, OperationSet]:
        # 获取二进制字符串
        binary1 = parent1.operations
        binary2 = parent2.operations

        # 选择交叉方法
        method = random.choice(['single_point', 'two_point', 'uniform'])

        if method == 'single_point':
            # 单点交叉
            point = random.randint(1, 16)
            child1_binary = binary1[:point] + binary2[point:]
            child2_binary = binary2[:point] + binary1[point:]

        elif method == 'two_point':
            # 两点交叉
            point1 = random.randint(1, 15)
            point2 = random.randint(point1 + 1, 16)
            child1_binary = binary1[:point1] + binary2[point1:point2] + binary1[point2:]
            child2_binary = binary2[:point1] + binary1[point1:point2] + binary2[point2:]

        else:  # uniform crossover
            # 均匀交叉
            child1_binary = list(binary1)
            child2_binary = list(binary2)
            for i in range(17):
                if random.random() < 0.5:
                    # 交换位
                    child1_binary[i], child2_binary[i] = child2_binary[i], child1_binary[i]
            child1_binary = ''.join(child1_binary)
            child2_binary = ''.join(child2_binary)

        # 创建子代
        child1 = OperationSet(child1_binary)
        child2 = OperationSet(child2_binary)

        return child1, child2
    def mutation(self, individual: OperationSet) -> OperationSet:

        # 获取当前二进制字符串
        binary_str = individual.operations

        # 确定变异位数
        n_mutations = random.randint(1, 5)

        # 将字符串转换为列表以便修改
        mutated_chars = list(binary_str)

        # 随机选择变异位置
        positions_to_mutate = random.sample(range(17), n_mutations)

        # 进行位翻转
        for pos in positions_to_mutate:
            # 位翻转：1变为0，0变为1
            mutated_chars[pos] = '1' if mutated_chars[pos] == '0' else '0'

        # 创建新的二进制字符串
        mutated_binary = ''.join(mutated_chars)

        # 返回新的OperationSet
        return OperationSet(mutated_binary)
    def aging_and_replacement(self):
        """老化与替换"""
        new_population = []

        for individual in self.population:
            individual.age += 1

            # 如果年龄太大 或者表现不佳，可能被淘汰
            if individual.age > self.max_age or individual.fitness < 0.5:
                pass
            else:
                new_population.append(individual)

        # 补充种群到指定大小
        while len(new_population) < self.population_size:
            new_ops = set(random.sample(range(1, 18), random.randint(2, 6)))
            new_population.append(OperationSet(new_ops))

        self.population = new_population[:self.population_size]

    def run_evolution(self, rm_samples: List[RMData],
                      transformer: JavaCodeTransformer,
                      llm_adapter: LLMAdapter,
                      sample_prompt: str = None) -> OperationSet:
        """
        运行遗传算法进化

        Returns:
            最佳操作集合
        """
        logger.info("开始遗传算法优化...")

        # 初始化种群
        self.initialize_population()

        # 初始评估
        logger.info(f"初始种群评估，每次使用 {self.eval_sample_size} 个样本...")
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = []
            for op_set in self.population:
                print('cxs2',type(op_set),op_set)
                futures.append(executor.submit(
                    self.evaluate_fitness, op_set, rm_samples,
                    transformer, llm_adapter
                ))

            for future in as_completed(futures):
                future.result()

        self.best_solution = max(self.population, key=lambda x: x.fitness)
        self.fitness_history.append(self.best_solution.fitness)

        logger.info(f"初始全局最佳F1: {self.best_solution.fitness:.4f}")

        # 进化循环
        stagnation_count = 0

        for generation in range(self.max_generations):
            logger.info(f"\n=== 第 {generation + 1} 代 ===")

            # 1. 老化与替换
            self.aging_and_replacement()
            # 2. 选择
            selected = self.selection()

            # 3. 交叉
            offspring = []
            while len (offspring)<self.population_size:
                for i in range(0, len(selected) - 1, 2):
                    child1, child2 = self.crossover(selected[i], selected[i+1])
                    offspring.extend([child1, child2])

            # 4. 变异
            offspring = [self.mutation(ind) for ind in offspring]

            # 5. 评估子代
            logger.info(f"评估子代，每次使用 {self.eval_sample_size} 个样本...")
            with ThreadPoolExecutor(max_workers=32) as executor:
                futures = []
                for op_set in offspring:
                    print('cxs3',type(op_set),op_set)
                    futures.append(executor.submit(
                        self.evaluate_fitness, op_set, rm_samples,
                        transformer, llm_adapter
                    ))

                for future in as_completed(futures):
                    future.result()

            # 6. 合并种群
            combined = self.population + offspring
            combined = sorted(combined, key=lambda x: x.fitness, reverse=True)
            self.population = combined[:self.population_size]
            # 7. 更新最佳解
            current_best = max(self.population, key=lambda x: x.fitness)
            self.fitness_history.append(current_best.fitness)

            # 检查改进
            if float(current_best.fitness)- float(self.best_solution.fitness)>0:
                self.best_solution = OperationSet(
                    operations=current_best.operations,
                    fitness=current_best.fitness,
                    age=0
                )
                stagnation_count = 0
                logger.info(f"发现新最佳! F1: {current_best.fitness:.4f}, 操作: {current_best.operations}")
            else:
                stagnation_count += 1
                logger.info(f"全局最佳F1：{self.best_solution.fitness}，当前最佳F1: {current_best.fitness:.4f}, 停滞: {stagnation_count}")

            # 检查终止条件
            if current_best.fitness >= self.target_f1:
                logger.info(f"达到目标F1: {self.target_f1}")
                break

            if stagnation_count >= self.stability_threshold:
                logger.info(f"连续 {stagnation_count} 代无改进，终止进化")
                break

            # 打印种群统计
            avg_fitness = sum(op.fitness for op in self.population) / len(self.population)
            logger.info(f"种群统计: 全局最佳F1：{self.best_solution.fitness}，种群平均F1={avg_fitness:.4f}, 种群最佳F1={current_best.fitness:.4f}")

        logger.info(f"\n进化完成! 最佳F1: {self.best_solution.fitness:.4f}")
        logger.info(f"最佳操作: {self.best_solution.operations}")

        return self.best_solution

    def visualize_evolution(self, output_dir: str = "evolution_results"):
        """可视化进化过程"""
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(12, 6))

        # F1分数变化
        plt.subplot(1, 2, 1)
        plt.plot(self.fitness_history, 'b-', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Best F1 Score')
        plt.title('Evolution of F1 Score')
        plt.grid(True, alpha=0.3)

        # 最终种群分布
        plt.subplot(1, 2, 2)
        fitness_values = [op.fitness for op in self.population]
        plt.hist(fitness_values, bins=10, alpha=0.7, color='green')
        plt.xlabel('F1 Score')
        plt.ylabel('Count')
        plt.title('Final Population Distribution')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'evolution_plot.png'), dpi=300)
        plt.show()

        # 保存进化历史
        history_df = pd.DataFrame({
            'generation': range(len(self.fitness_history)),
            'best_f1': self.fitness_history
        })
        history_df.to_csv(os.path.join(output_dir, 'evolution_history.csv'), index=False)

class RMDataLoader:
    """RM数据加载器"""

    @staticmethod
    def load_rm_data(seqs: List[int],
                     av_agent_output_dir: str,
                     av_agent_final_dir: str) -> List[RMData]:
        """
        加载RM数据

        Args:
            seqs: 序列号列表
            av_agent_output_dir: av_agent_output目录
            av_agent_final_dir: av_agent_final_4_4目录

        Returns:
            RM数据列表
        """
        rm_data_list = []
        conn=get_connection()
        for seq in seqs:
            seq=str(seq)
            try:
                # 从av_agent_output加载模板
                template_file = os.path.join(av_agent_output_dir, f'seq_{seq}_template.json')
                if os.path.exists(template_file):
                    with open(template_file, 'r', encoding='utf-8') as f:
                        template_data = json.load(f)

                    # 获取RM内容
                    rm_content = template_data.get('RM', '')

                    # 从av_agent_final_4_4获取预测结果
                    result_file = os.path.join(av_agent_final_dir, 'av_agent_final_results.json')
                    if os.path.exists(result_file):
                        with open(result_file, 'r', encoding='utf-8') as f:
                            results_data = json.load(f)

                        if seq in results_data:
                            result = results_data[seq]
                            predicted = 1 if result.get('final_classification') == 'Malicious' else 0
                            confidence = float(result.get('confidence_scores', 0.5))

                            # 获取真实标签（从数据库或现有数据）
                            true_label = result.get('true_label',
                                                    RMDataLoader._get_true_label_from_db(conn,seq))

                            # 创建RM文件路径
                            rm_file = f"RM_{seq}.java"

                            rm_data = RMData(
                                seq=seq,
                                content=rm_content,
                                true_label=true_label,
                                predicted_label=predicted,
                                confidence=confidence,
                                file_path=rm_file
                            )
                            rm_data_list.append(rm_data)

            except Exception as e:
                logger.error(f"加载序列 {seq} 数据失败: {e}")
                continue
        conn.close()
        logger.info(f"成功加载 {len(rm_data_list)} 个RM样本")
        return rm_data_list

    @staticmethod
    def _get_true_label_from_db(conn,seq: str) -> int:
        label=0
        with conn.cursor() as cursor:
            sql = "SELECT label FROM app_label WHERE seq = %s"
            cursor.execute(sql, (seq,))
            label_result = cursor.fetchone()
            label=0 if label_result[0] == 'B' else 1
        return label
    @staticmethod
    def sample_rm_data(rm_data_list: List[RMData], sample_size: int = 20,
                       balance: bool = True) -> List[RMData]:
        """采样RM数据"""
        if not balance or len(rm_data_list) <= sample_size:
            return random.sample(rm_data_list, min(sample_size, len(rm_data_list)))

        # 按标签分组
        malicious = [d for d in rm_data_list if d.true_label == 1]
        benign = [d for d in rm_data_list if d.true_label == 0]

        # 平衡采样
        n_mal = min(len(malicious), sample_size // 2)
        n_ben = min(len(benign), sample_size // 2)

        sampled = random.sample(malicious, n_mal) + random.sample(benign, n_ben)

        # 如果还需要更多样本
        if len(sampled) < sample_size:
            remaining = [d for d in rm_data_list if d not in sampled]
            sampled.extend(random.sample(remaining, min(sample_size - len(sampled), len(remaining))))

        return sampled
def get_operation_details(best_operations):
    """获取二进制字符串对应的操作详情"""
    binary_str = best_operations.operations

    # 确保是17位二进制字符串
    if len(binary_str) != 17:
        binary_str = binary_str[:17].ljust(17, '0')

    # 操作描述字典
    operation_descriptions = {
        1: "移除catch块",
        2: "移除变量声明",
        3: "移除函数调用",
        4: "移除条件判断",
        5: "移除异常抛出",
        6: "移除注释内容",
        7: "移除访问修饰符",
        8: "移除循环控制",
        9: "移除导入语句",
        10: "移除运算符（算术、关系、逻辑）",
        11: "移除字面量（数字、字符串、布尔）",
        12: "移除分隔符（括号、分号、逗号）",
        13: "移除空白符（空格、制表、换行）",
        14: "移除参数（形参、实参）",
        15: "移除函数名字（调用、声明、创建新对象）",
        16: "移除类型名字（变量的类型）",
        17: "移除关键字（java预设的）"
    }

    # 找出所有为1的位对应的操作
    active_operations = []
    for i in range(17):
        if binary_str[i] == '1':
            op_num = i + 1
            if op_num in operation_descriptions:
                active_operations.append((op_num, operation_descriptions[op_num]))

    return active_operations

def main():
    """主函数"""

    # 配置参数
    SEQS = [11071,11119,11135,11186,11188,11271,11346,11367,11395,11399,11413,11421,11436,11473,11486,11494,11499,11529,11544,11573,11586,11596,11623,11676,11682,11692,11707,11743,11753,11764,11827,11834,11842,11853,11865,11877,11908,11940,12008,12071,12096,12104,12135,12137,12140,12144,12145,12154,12161,12163,54346,54374,54560,54640,55030,55161,55373,55392,55485,55686,55975,55984,56185,56267,56271,56424,56488,56556,56750,57205,57363,57571,57601,57667,57735,58107,58945,58971,58984,59081,59125,59352,59431,59652,59826,60021,60126,60140,60789,60918,61087,61274,61479,61507,61530,61737,61786,61926,61942,61997]  # 示例序列号，实际需要更多
    AV_AGENT_OUTPUT_DIR = "av_agent_output_mu"
    AV_AGENT_FINAL_DIR = "av_agent_final_mu"
    # 遗传算法参数
    POPULATION_SIZE = 4#10#指令数量
    MAX_GENERATIONS = 20#50#最大遗传代数
    TARGET_F1 = 0.95   #目标分数
    EVAL_SAMPLE_SIZE = 100#每次评估选择的样本数量
    EVAL_SAMPLE_RATIO = 0.5  #每次采样正负样本比

    # 初始化组件
    logger.info("初始化组件...")

    # 1. 加载RM数据
    loader = RMDataLoader()
    all_rm_data = loader.load_rm_data(SEQS, AV_AGENT_OUTPUT_DIR, AV_AGENT_FINAL_DIR)

    if not all_rm_data:
        logger.error("未加载到RM数据，请检查路径和序列号")
        return

    # 2. 初始化转换器和LLM适配器
    transformer = JavaCodeTransformer()
    llm_adapter = LLMAdapter()

    # 4. 初始化遗传算法优化器
    optimizer = GeneticOptimizer(
        population_size=POPULATION_SIZE,
        max_generations=MAX_GENERATIONS,
        target_f1=TARGET_F1,
        eval_sample_size=EVAL_SAMPLE_SIZE,
        eval_sample_ratio=EVAL_SAMPLE_RATIO
    )

    # 5. 运行优化
    best_operations = optimizer.run_evolution(
        rm_samples=all_rm_data,
        transformer=transformer,
        llm_adapter=llm_adapter
    )

    # 6. 可视化结果
    optimizer.visualize_evolution()

    # 7. 保存最佳操作集合
    output_dir = "optimization_results"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'best_operations.json'), 'w') as f:
        json.dump({
            'operations': best_operations.operations,
            'f1_score': best_operations.fitness,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)

    # 8. 在测试集上验证
    # logger.info("\n=== 在测试集上验证最佳操作 ===")
    #
    # # 采样测试集
    #
    # test_f1 = optimizer.evaluate_fitness(
    #     best_operations, all_rm_data, transformer, llm_adapter
    # )
    #
    # logger.info(f"总的样本的 F1分数: {test_f1:.4f}")

    # 9. 生成操作说明
    active_ops = get_operation_details(best_operations)
    opt=[]
    for op_num, description in active_ops:
        opt.append(description)
    logger.info("\n=== 最佳操作 ===")
    logger.info(f"操作 {best_operations.operations}: {opt}")
    logger.info(f"全局最佳 F1: {best_operations.fitness:.4f}")
    # logger.info(f"训练F1: {best_operations.fitness:.4f}, 测试F1: {test_f1:.4f}")

if __name__ == "__main__":
    main()