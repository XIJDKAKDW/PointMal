# coding=utf-8
#1、得到基模型	1
import logging
import os
import platform
import tempfile

import joblib
import numpy as np
import pymysql
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer as TF
from torch.utils.data import Dataset
import sklearn as sk
from sklearn.metrics import balanced_accuracy_score

system = platform.system()

# 设置日志
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
def plot_value(y_true, y_pred):
    balance_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = sk.metrics.precision_score(y_true, y_pred)
    recall = sk.metrics.recall_score(y_true, y_pred)
    f1_value = sk.metrics.f1_score(y_true, y_pred)
    return balance_accuracy, precision, recall, f1_value
# 将lambda函数定义为独立的函数，以便序列化
def tokenizer_func(x):
    """Tokenizer function for TF-IDF vectorizer"""
    return x.split('\n')

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
        port = 3306,
    )
    return conn

# 3. 传统机器学习模型 (用于Drebin特征)
class DrebinModel:
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        self.vectorizer = None

    def fit(self, file_paths, y):
        # 使用TF-IDF向量化文件路径
        self.vectorizer = TF(
            input="filename",
            tokenizer=tokenizer_func,
            token_pattern=None,
            binary=True
        )
        X = self.vectorizer.fit_transform(file_paths)
        self.model.fit(X, y)
        return self

    def predict_proba(self, file_paths):
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted")
        X = self.vectorizer.transform(file_paths)
        return self.model.predict_proba(X)

# 数据集类 - 对齐论文中的三种特征
class RawBytesDataset(Dataset):
    def __init__(self, seqs, conn):
        self.seqs = seqs
        self.conn = conn
        self.data = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        """从s3Feature表加载原始字节特征"""
        for seq in self.seqs:
            try:
                with self.conn.cursor() as cursor:
                    sql = "SELECT feature FROM s3_feature WHERE apkSeq = %s"
                    cursor.execute(sql, (seq,))
                    result = cursor.fetchone()

                    if result and result[0]:
                        feature_vector = [float(x.replace('[','').replace(']','')) for x in result[0].split(',')]
                        self.data.append(feature_vector)
                        label = self._get_label_by_seq(seq)
                        self.labels.append(label)
            except Exception as e:
                logger.error(f"Error loading raw bytes for seq {seq}: {e}")
                continue

    def _get_label_by_seq(self, seq):
        """根据seq获取标签"""
        try:
            with self.conn.cursor() as cursor:
                sql = "SELECT label FROM app_label WHERE seq = %s"
                cursor.execute(sql, (seq,))
                result = cursor.fetchone()
                if result:
                    return 0 if result[0] == 'B' else 1
        except Exception as e:
            logger.error(f"Error getting label for seq {seq}: {e}")
        return 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.tensor(self.labels[idx], dtype=torch.long)

class DrebinFeatureDataset:
    def __init__(self, seqs, conn, is_training=True, vectorizer=None):
        self.seqs = seqs
        self.conn = conn
        self.file_paths = []
        self.labels = []
        self.is_training = is_training
        self.vectorizer = vectorizer
        self._load_data()

    def _load_data(self):
        """从drebin_feature表加载文件路径"""
        for seq in self.seqs:
            try:
                with self.conn.cursor() as cursor:
                    sql = "SELECT path FROM drebin_feature WHERE apkSeq = %s"
                    cursor.execute(sql, (seq,))
                    result = cursor.fetchone()

                    if result and result[0]:
                        file_path = result[0]
                        if system == "Linux":
                            pass
                        else:
                            file_path = file_path.replace('/home/changxiaosong/dataset', r'D:')
                        self.file_paths.append(file_path)

                        label = self._get_label_by_seq(seq)
                        self.labels.append(label)
                        #logger.info(f"Loaded Drebin seq {seq}: file_path={file_path}, label={label}")
            except Exception as e:
                logger.error(f"Error loading Drebin feature for seq {seq}: {e}")
                continue

    def _get_label_by_seq(self, seq):
        """根据seq获取标签"""
        try:
            with self.conn.cursor() as cursor:
                sql = "SELECT label FROM app_label WHERE seq = %s"
                cursor.execute(sql, (seq,))
                result = cursor.fetchone()
                if result:
                    return 0 if result[0] == 'B' else 1
        except Exception as e:
            logger.error(f"Error getting label for seq {seq}: {e}")
        return 0

    def get_features_and_labels(self):
        """返回特征和标签"""
        if self.is_training:
            # 训练模式：创建并拟合TF-IDF向量化器
            self.vectorizer = TF(
                input="filename",
                tokenizer=tokenizer_func,
                token_pattern=None,
                binary=True
            )
            X = self.vectorizer.fit_transform(self.file_paths)
        else:
            # 测试模式：使用现有的向量化器转换数据
            if self.vectorizer is not None:
                X = self.vectorizer.transform(self.file_paths)
            else:
                logger.error("TF-IDF vectorizer not provided for test data")
                return np.array([]), np.array([])

        #logger.info(f"Drebin TF-IDF特征矩阵形状: {X.shape}")
        return X, np.array(self.labels)

    def get_vectorizer(self):
        """返回TF-IDF向量化器"""
        return self.vectorizer

    def get_file_paths(self):
        """返回文件路径"""
        return self.file_paths

def train_drebin_model(train_file):
    """训练Drebin特征分类器"""
    logger.info("开始训练Drebin特征分类器...")

    seqs_train = load_seqs_from_file(train_file)
    conn = get_connection()
    if conn is None:
        return None, None

    dataset = DrebinFeatureDataset(seqs_train, conn, is_training=True)
    X_train, y_train = dataset.get_features_and_labels()
    vectorizer = dataset.get_vectorizer()

    if X_train.shape[0] == 0:
        logger.error("错误: 没有有效的Drebin特征数据")
        conn.close()
        return None, None

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)

    conn.close()

    # 保存模型和向量化器
    joblib.dump(model, 'drebin_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

    logger.info("Drebin特征分类器训练完成")
    return model, vectorizer

def generate_detection_thresholds(train_file, seqs_test):
    """生成三类检测阈值并存入数据库 - 对齐论文的层次特征构建"""
    pre_dict={}
    logger.info("开始生成检测阈值...")
    # 训练三个模型 - 对应论文中的层次特征
    drebin_model, vectorizer = train_drebin_model(train_file)

    if drebin_model is None :
        logger.error("模型训练失败，无法生成检测阈值")
        return

    conn = get_connection()
    if conn is None:
        return
    try:
        with conn.cursor() as cursor:
            for seq in seqs_test:
                try:
                    # 获取Drebin特征预测概率
                    pro_drebin = predict_drebin_probability(drebin_model, vectorizer, seq, conn)
                    pre_dict[seq]=pro_drebin
                except Exception as e:
                    logger.error(f"处理seq {seq}时出错: {e}")
                    continue

        logger.info("检测阈值生成完成并存入数据库")

    except Exception as e:
        logger.error(f"生成检测阈值时出错: {e}")
        conn.rollback()
    finally:
        conn.close()
        return pre_dict

def predict_drebin_probability(model, vectorizer, seq, conn):
    """预测Drebin特征模型的恶意概率"""
    with conn.cursor() as cursor:
        sql = "SELECT path FROM drebin_feature WHERE apkSeq = %s"
        cursor.execute(sql, (str(seq),))
        result = cursor.fetchone()

        if result and result[0]:
            file_path = result[0]
            if system == "Linux":
                pass
            else:
                file_path = file_path.replace('/home/changxiaosong/dataset', r'D:')

            # 使用TF-IDF向量化器转换特征
            features_tfidf = vectorizer.transform([file_path])
            # 预测概率
            probabilities = model.predict_proba(features_tfidf)
            malicious_prob = probabilities[0, 1]
            return f"{malicious_prob:.4f}"


def predict_using_feature(model, vectorizer,seq,feature_has ):
    """预测Drebin特征模型的恶意概率"""
    feature_tmp = str(seq) + r'_tmp.txt'
    with open(feature_tmp, 'w', newline='\n') as f:
        f.write('\n'.join(feature_has))
    features_tfidf = vectorizer.transform([feature_tmp])
    # 删除文件
    probabilities = model.predict_proba(features_tfidf)
    malicious_prob = probabilities[0, 1]
    os.remove(feature_tmp)
    return f"{malicious_prob:.4f}"

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
        logger.error(f"错误: 加载文件时出错: {e}")
    return seqs
def get_label_by_seq(conn, seq):
    """根据seq获取标签"""
    try:
        with conn.cursor() as cursor:
            sql = "SELECT label FROM app_label WHERE seq = %s"
            cursor.execute(sql, (seq,))
            result = cursor.fetchone()
            if result:
                return 0 if result[0] == 'B' else 1
            else:
                return 0  # 默认返回良性
    except Exception as e:
        logger.error(f"Error getting label for seq {seq}: {e}")
        return 0
def main(train_file ,seqs_test):
    logger.info("开始训练模型并生成检测阈值...")
    pre_dict=generate_detection_thresholds(train_file, seqs_test)
    true_labels = []  # 存储真实标签
    final_predictions = []  # 存储最终预测
    conn = get_connection()
    for one in pre_dict:
        final_predictions.append(0 if pre_dict[one] is None else 1  if float(pre_dict[one]) >= 0.5 else 0)
        label = get_label_by_seq(conn, one)
        true_labels.append(label)
    conn.close()
    balance_accuracy_llm, precision_llm, recall_llm, f1_value_llm = \
        plot_value(true_labels, final_predictions)
    print(['llm-result 基模型性能：',balance_accuracy_llm, precision_llm, recall_llm, f1_value_llm])
    return pre_dict
def main_abla(train_file ,seqs_test):
    logger.info("开始训练模型并生成检测阈值...")
    pre_dict=generate_detection_thresholds(train_file, seqs_test)
    true_labels = []  # 存储真实标签
    final_predictions = []  # 存储最终预测
    conn = get_connection()
    for one in pre_dict:
        final_predictions.append(0 if pre_dict[one] is None else 1  if float(pre_dict[one]) >= 0.5 else 0)
        label = get_label_by_seq(conn, one)
        true_labels.append(label)
    conn.close()
    balance_accuracy_llm, precision_llm, recall_llm, f1_value_llm = \
        plot_value(true_labels, final_predictions)
    return pre_dict,balance_accuracy_llm, precision_llm, recall_llm, f1_value_llm
