#2、得到可疑特征	1
#3、获取可疑区域	1
import json
import logging
import os
import pickle
import platform
import sys
import threading
from typing import Dict, List, Tuple

import joblib
import numpy as np
import shap
import torch

system = platform.system()

if system == "Linux":
    sys.path.append(r"/home/changxiaosong/python/malwareTest")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr2")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr2_new")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr2_final")
from pr2_final.test001 import predict_drebin_probability
from test001Method_new_9_4_3 import GroupTestingNodeSelector
from test001Method_new_9_4_3 import get_file_path_by_seq, analyze_risk_components, clear

from combine_compare_tool_method import get_connection

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def load_sensitive_graph(graph_path):
    try:
        if not os.path.exists(graph_path):
            return None

        with open(graph_path, 'rb') as f:
            tmp=pickle.load(f)
            graph = tmp[0] if isinstance(tmp,tuple) else tmp
        return graph
    except Exception as e:
        print(graph_path,f"加载图时出错: {e}")
        return None

def init_shap_models(drebin_model, feature_vectorizer):
    selector = GroupTestingNodeSelector(
        seq=None,
        drebin_model=drebin_model,
        feature_vectorizer=feature_vectorizer,
        contrastive_model_path='visualizations/contrastive_model.pth',
        deepseek_model_path='/home/changxiaosong/python/malwareTest/deepseek-coder-1.3b-base'
    )
    # 初始化SHAP解释器
    explainer = shap.TreeExplainer(drebin_model)
    feature_names = feature_vectorizer.get_feature_names_out()
    print("特征向量化和提取模型初始化完成")
    return selector, feature_names, explainer
def tokenizer_func(x):
    """Tokenizer function for TF-IDF vectorizer"""
    return x.split('\n')

def get_feature_file_label_by_seq(conn, seqs):
    """根据序列号获取文件路径和标签，只返回有效样本"""
    file_paths = []
    labels = []
    valid_seqs = []  # 记录有效的seq，用于后续对应

    for seq in seqs:
        try:
            with conn.cursor() as cursor:
                # 获取标签
                sql = "SELECT label FROM app_label WHERE seq = %s"
                cursor.execute(sql, (seq,))
                label_result = cursor.fetchone()

                if label_result:
                    label = 0 if label_result[0] == 'B' else 1
                else:
                    label = 0
                labels.append(label)
                # 获取文件路径
                sql = "SELECT path FROM drebin_feature WHERE apkSeq = %s"
                cursor.execute(sql, (seq,))
                result = cursor.fetchone()
                file_path=''
                if result and result[0]:  # 只有找到有效路径才处理
                    file_path = result[0]
                    if platform.system() != "Linux":
                        file_path = file_path.replace('/home/changxiaosong/dataset', r'D:')
                file_paths.append(file_path)

        except Exception as e:
            logger.error(f"获取序列 {seq} 的文件路径和标签失败: {e}")
            continue

    return file_paths, labels
class FeatureEngineering:
    """特征工程模块 - 实现论文中的字符特征提取"""

    def __init__(self, drebin_model, feature_vectorizer, device=None):
        """
        初始化特征工程模块
        drebin_model: 训练好的Drebin GBDT模型
        feature_vectorizer: TF-IDF向量化器
        """
        self.drebin_model = drebin_model
        self.feature_vectorizer = feature_vectorizer
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def extract_key_features_with_shap(self, file_paths: List[str], labels: List[int], top_k: int = 10) -> Dict:
        """
        使用SHAP提取关键特征
        返回: 包含关键特征信息的字典
        """
        logger.info("使用SHAP提取关键特征...")

        # 过滤掉空字符串和None值
        filtered_paths = []
        filtered_labels = []

        for i, (file_path, label) in enumerate(zip(file_paths, labels)):
            # 检查是否为None或空字符串（包括只包含空格的字符串）
            if file_path is None or not isinstance(file_path, str) or not file_path.strip():
                logger.warning(f"跳过无效文件路径: '{file_path}' (索引: {i})")
                continue
            filtered_paths.append(file_path.strip())
            filtered_labels.append(label)

        logger.info(f"原始文件数: {len(file_paths)}, 有效文件数: {len(filtered_paths)}")

        # 转换特征 - 保持为稀疏矩阵格式
        X = self.feature_vectorizer.transform(filtered_paths)

        feature_names = self.feature_vectorizer.get_feature_names_out()

        # 创建SHAP解释器
        explainer = shap.TreeExplainer(self.drebin_model)

        # 将数据转换为适合SHAP的格式
        if hasattr(X, "toarray"):
            X_array = X.toarray()
        else:
            X_array = X

        # 计算SHAP值
        try:
            shap_values = explainer.shap_values(X_array)
        except Exception as e:
            # 记录关键信息
            logger.error(f"SHAP值计算失败: {str(e)}")
            logger.error(f"错误类型: {type(e).__name__}")
            logger.error(f"X_array形状: {X_array.shape}")
            logger.error(f"X_array数据类型: {X_array.dtype}")
            logger.error(f"X_array非零元素数: {np.count_nonzero(X_array)}")
            logger.error(f"X_array最小值: {np.min(X_array)}, 最大值: {np.max(X_array)}")
            logger.error(f"样本数量: {len(filtered_paths)}")
            logger.error(f"特征数量: {len(feature_names)}")

            # 原样抛出异常
            raise

        key_features = {
            'global_importance': self._get_global_feature_importance(explainer, X_array, feature_names),
            'sample_specific': {}
        }

        # 为每个样本提取关键特征
        for i, (file_path, label) in enumerate(zip(filtered_paths, filtered_labels)):
            sample_key_features = self._get_sample_key_features(
                shap_values, X_array, i, feature_names, top_k
            )
            key_features['sample_specific'][file_path] = {
                'key_features': sample_key_features,
                'true_label': label,
                'predicted_label': self.drebin_model.predict(X_array[i:i+1])[0],
                'confidence': np.max(self.drebin_model.predict_proba(X_array[i:i+1]))
            }

        return key_features

    def _get_sample_key_features(self, shap_values, X_array, sample_idx: int,
                                 feature_names: List[str], top_k: int = 10) -> List[Dict]:
        """
        获取单个样本的关键特征
        """
        # 处理多分类情况
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]  # 使用恶意类别的SHAP值
        else:
            shap_vals = shap_values

        # 获取当前样本的SHAP值
        sample_shap = shap_vals[sample_idx]

        # 获取特征值
        if hasattr(X_array, "iloc"):
            feature_values = X_array.iloc[sample_idx].values
        else:
            feature_values = X_array[sample_idx]

        # 创建特征重要性列表
        feature_importance = []
        for j in range(len(feature_names)):
            if feature_values[j] != 0:  # 只考虑存在的特征
                shap_val = sample_shap[j]
                feature_importance.append({
                    'feature_name': feature_names[j],
                    'shap_value': float(shap_val),
                    'feature_value': float(feature_values[j]),
                    'feature_type': self._categorize_feature(feature_names[j]),
                    'semantic_description': self._get_semantic_description(feature_names[j])
                })

        # 按SHAP值的绝对值排序，取前top_k个
        feature_importance.sort(key=lambda x: x['shap_value'], reverse=True)
        return feature_importance[:top_k]

    def _get_global_feature_importance(self, explainer, X_array, feature_names, top_k: int = 20) -> List[Tuple]:
        """获取全局特征重要性"""
        # 使用模型自带的特征重要性
        if hasattr(self.drebin_model, 'feature_importances_'):
            importances = self.drebin_model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_k]
            return [(feature_names[i], float(importances[i])) for i in indices]

        # 备用方法：使用SHAP的均值绝对值
        try:
            shap_vals = explainer.shap_values(X_array)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]  # 恶意类别
            mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
            indices = np.argsort(mean_abs_shap)[::-1][:top_k]
            return [(feature_names[i], float(mean_abs_shap[i])) for i in indices]
        except Exception as e:
            logger.error(f"全局特征重要性计算失败: {e}")
            return []

    def _categorize_feature(self, feature_name: str) -> str:
        """对Drebin特征进行分类"""
        feature_lower = feature_name.lower()

        if 'permission' in feature_lower:
            return 'permission'
        elif 'api' in feature_lower:
            return 'api'
        elif 'activity' in feature_lower:
            return 'activity'
        elif 'service' in feature_lower:
            return 'service'
        elif 'receiver' in feature_lower:
            return 'receiver'
        elif 'provider' in feature_lower:
            return 'provider'
        elif 'intent' in feature_lower:
            return 'intent'
        elif 'url' in feature_lower or 'domain' in feature_lower:
            return 'network'
        elif 'hardware' in feature_lower:
            return 'hardware'
        else:
            return 'other'

    def _get_semantic_description(self, feature_name: str) -> str:
        """Get semantic description of the feature"""
        feature_lower = feature_name.lower()

        # Permission-related features
        if 'permission' in feature_lower:
            perm_name = feature_name.split('permissionslist_')[-1] if 'permissionslist_' in feature_lower else feature_name
            return f"Requested permission: {perm_name}"

        # API-related features
        elif 'api' in feature_lower:
            api_name = feature_name.split('apilist_')[-1] if 'apilist_' in feature_lower else feature_name
            return f"API call: {api_name}"

        # Component-related features
        elif any(comp in feature_lower for comp in ['activity', 'service', 'receiver', 'provider']):
            comp_type = 'Activity' if 'activity' in feature_lower else \
                'Service' if 'service' in feature_lower else \
                    'BroadcastReceiver' if 'receiver' in feature_lower else 'ContentProvider'
            comp_name = feature_name.split('list_')[-1]
            return f"{comp_type} component: {comp_name}"

        # Network-related features
        elif 'url' in feature_lower or 'domain' in feature_lower:
            url = feature_name.split('urldomainlist_')[-1] if 'urldomainlist_' in feature_lower else feature_name
            return f"Network connection: {url}"

        else:
            return f"Feature: {feature_name}"
class LLMFeatureFormatter:
    """格式化特征用于LLM推理"""

    @staticmethod
    def format_confidence_scores(confidence_scores: Dict) -> str:
        """Format confidence scores"""
        formatted = "Classifier Confidence Scores:\n"
        for feature_type, scores in confidence_scores.items():
            formatted += f"- {feature_type}: {scores['confidence']:.4f} (Prediction: {'Malicious' if scores['predicted_class'] == 1 else 'Benign'})\n"
        return formatted
    @staticmethod
    def format_key_features(key_features: Dict) -> str:
        """Format key features"""
        formatted = "Key Semantic Feature Analysis:\n"

        # Group features by type
        features_by_type = {}
        for feature_info in key_features:
            feature_type = feature_info['feature_type']
            if feature_type not in features_by_type:
                features_by_type[feature_type] = []
            features_by_type[feature_type].append(feature_info)

        # Output features by type
        for feature_type, features in features_by_type.items():
            formatted += f"\n{feature_type.upper()} FEATURES:\n"
            for feat in features[:5]:  # Show max 5 features per type
                influence = "Positive" if feat['shap_value'] > 0 else "Negative"
                formatted += f"  • {feat['semantic_description']} (Influence: {influence}, Strength: {abs(feat['shap_value']):.4f})\n"

        return formatted


class ThreadOwn(threading.Thread):
    def __init__(self, func, args=()):
        super(ThreadOwn, self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.result = self.func(*self.args)
    def get_result(self):
        threading.Thread.join(self)  # 等待线程执行完毕
        try:
            return self.result
        except Exception:
            return None


from concurrent.futures import ThreadPoolExecutor, as_completed
lock_dnn = threading.Lock()
def process_single_sample(args):
    """处理单个样本的函数，用于多线程"""
    i, seq, data = args
    drebin_model, feature_vectorizer, conn, llm_formatter, file_paths, labels, key_features_data, \
    system, smali_tmp, baksmali_path, jadx_path, smali_path, \
    graph_dir, android_sdk, per_api_dir, soot_jar, SourcesAndSinks, taint_output_dir,selector,feature_names,explainer = data
    # 获取该样本的置信度分数
    with lock_dnn:
        confidence_scores = predict_drebin_probability(drebin_model, feature_vectorizer, seq, conn)
    confidence_scores = {
        "drebin": {
            "confidence": confidence_scores,
            "predicted_class":0 if confidence_scores is None else 1 if float(confidence_scores) >= 0.5 else 0
        }
    }

    # 获取该样本的关键特征
    file_path = file_paths[i]
    if file_path in key_features_data['sample_specific']:
        sample_features = key_features_data['sample_specific'][file_path]['key_features']
    else:
        sample_features = []

    # 获取可疑区域
    apkName, apkPath = get_file_path_by_seq(seq)
    if system == "Linux":
        apkPath = apkPath.replace('F:/malware-app', '/home/changxiaosong/dataset')
    else:
        apkPath = apkPath.replace('F:/malware-app', 'E:/malware-app')
    ApkFile = os.path.abspath(apkPath)
    graph_file = graph_dir + os.sep + str(seq) + '.pkl'

    graph = None
    if os.path.exists(graph_file):
        graph = load_sensitive_graph(graph_file)
    # if graph is None:
    #     # 得到图
    #     if system == "Linux":
    #         thread_num = 32
    #     else:
    #         thread_num = 1
    #
    #     # 得到分析工具
    #     smali_dir, smali_status = get_smali(smali_tmp, baksmali_path, apkPath, apkName)
    #
    #     if smali_status:
    #         try:
    #             smaliAnalyzer = SmaliAnalyzer(smali_dir)
    #             apkAnalyzer = ApkAnalyzer(ApkFile, smali_dir)
    #         except Exception as e:
    #             print(f'代码行 {inspect.currentframe().f_lineno}')
    #             print(e)
    #             return seq, None
    #
    #         api2Permission = get_sensitive_apis_extend(per_api_dir)
    #         graph = generate_apk_method_graph(smali_dir, api2Permission, apkAnalyzer, smaliAnalyzer, thread_num)
    #
    #         if graph is not None:
    #             a = len(graph.edges())
    #             graph_flowid = analyze_call_graph(android_sdk, soot_jar, seq, apkPath)
    #             add_edges_from_dict(graph, graph_flowid)
    #             b = len(graph.edges())
    #             # graph_flowid = analyze_date_dependency(android_sdk, apkPath, taint_output_dir, soot_jar, SourcesAndSinks)
    #             # add_edges_from_dict(graph, graph_flowid)
    #             c = len(graph.edges())
    #             print(seq, '图的边数数量变化', a, b, c)
    #             save_graph(graph, graph_file)

    risk_nodes = []
    if graph is not None:
        # ============= 组测试节点选择方法（使用预训练嵌入） =============
        try:
            # 1. 获取测试样本的特征向量用于SHAP计算
            conn_local = get_connection()
            with conn_local.cursor() as cursor:
                sql = "SELECT path FROM drebin_feature WHERE apkSeq = %s"
                cursor.execute(sql, (str(seq),))
                result = cursor.fetchone()
                if result and result[0]:
                    file_path_tmp = result[0]
                    if system == "Linux":
                        pass
                    else:
                        file_path_tmp = file_path_tmp.replace('/home/changxiaosong/dataset', r'D:')

                    features_tfidf = feature_vectorizer.transform([file_path_tmp])
                    x_sample_np = features_tfidf.toarray()
                else:
                    x_sample_np = None
            conn_local.close()
            if x_sample_np is not None:
                with lock_dnn:
                    selector.reset_state(seq)
                    risk_nodes = selector.select_suspicious_nodes(
                        graph=graph,
                        explainer=explainer,
                        feature_names=feature_names,
                        x_sample_np=x_sample_np
                    )
                    # print(f'cxs 组测试选择可疑节点数量: {len(risk_nodes)}')
                # vis_path = f"final_selected_nodes_seq_{seq}.png"
                # selector.visualize_selected_nodes(vis_path)
        except Exception as e:
            print(f'组测试节点选择失败: {e}')
    RM_str=''
    for one in risk_nodes:
        RM_str_one, _, _, _ = analyze_risk_components(seq, [one], jadx_path, smali_path)
        RM_str_one, _, _, _ = clear(RM_str_one, '', '', '')
        RM_str=RM_str+'@@@cxs@@@'+RM_str_one
    result = {
        'confidence_scores': confidence_scores,
        'key_features': sample_features,
        'RM_str': RM_str,
        'AC_str': '',
        'AI_str': '',
        'CM_str': '',
        'true_label': labels[i],
        'file_path': file_path
    }

    return seq, result


def extract_features_for_llm(seqs: List[int], drebin_model, feature_vectorizer,
                             conn, llm_formatter: LLMFeatureFormatter = None,
                             max_workers: int = None) -> Dict:
    """
    完整的特征工程流程：从序列号提取特征并格式化为LLM输入
    多线程版本
    """

    # 获取文件路径和标签
    file_paths, labels  = get_feature_file_label_by_seq(conn, seqs)
    # 初始化特征工程
    feature_engineer = FeatureEngineering(drebin_model, feature_vectorizer)

    results = {}
    smali_tmp = r'..\pr2\decompiled_java\smali_tmp'
    baksmali_path = 'D:\\dexCompile\\program\\baksmali-2.5.2.jar'
    jadx_path = r"D:\jadx\bin\jadx.bat"
    smali_path = r'D:\\dexCompile\\program\\smali-2.5.2.jar'
    graph_dir = r'../pr2/decompiled_java/graph_tmp'
    android_sdk = r'D:/androidSDK/platforms'
    per_api_dir = r'../pr2/sensitive_api_list'
    soot_jar = r'D:\GitHub-download-package\CallGraph-Flowdroid-master\soot-infoflow-cmd-jar-with-dependencies.jar'
    SourcesAndSinks = r'..\SourcesAndSinks.txt'
    taint_output_dir = r"..\taint_output"

    if system == "Linux":
        soot_jar = r'/home/changxiaosong/python/malwareTest/pr2/soot-infoflow-cmd-jar-with-dependencies.jar'
        android_sdk = '/home/changxiaosong/python/obfuscapk/platforms'
        smali_tmp = '/home/changxiaosong/python/malwareTest/pr2' + os.sep + 'decompiled_java' + os.sep + 'smali_tmp'
        baksmali_path = '/home/changxiaosong/dexCompile/program/baksmali-2.5.2.jar'
        jadx_path = r"/home/changxiaosong/jadx/bin/jadx"
        smali_path = r'/home/changxiaosong/python/malwareTest/smali-2.5.2.jar'
        # graph_dir = '/home/changxiaosong/python/malwareTest/pr2' + os.sep + \
        #             'decompiled_java' + os.sep + 'graph_tmp'+ os.sep +'gra_old'
        graph_dir =r'/home/changxiaosong/python/malwareTest/pr2/decompiled_java/graph_tmp/graph_only_sensitive'

        per_api_dir = '/home/changxiaosong/python/malwareTest/pr2' + os.sep + 'sensitive_api_list'
        SourcesAndSinks = '/home/changxiaosong/python/malwareTest' + os.sep + 'SourcesAndSinks.txt'
        taint_output_dir = '/home/changxiaosong/python/malwareTest/pr2' + os.sep + "taint_output"

    # 提取关键特征（这部分可能需要保留单线程，因为依赖关系）
    key_features_data = feature_engineer.extract_key_features_with_shap(file_paths, labels)
    selector,feature_names,explainer=init_shap_models(drebin_model, feature_vectorizer)

    # 准备线程池
    if max_workers is None:
        # 根据系统自动设置线程数
        max_workers = 32 if system == "Linux" else 1

    # 准备任务参数
    shared_data = (
        drebin_model, feature_vectorizer, conn, llm_formatter,
        file_paths, labels, key_features_data,
        system, smali_tmp, baksmali_path, jadx_path, smali_path,
        graph_dir, android_sdk, per_api_dir, soot_jar, SourcesAndSinks, taint_output_dir,selector,feature_names,explainer
    )

    tasks = [(i, seq, shared_data) for i, seq in enumerate(seqs)]
    # 使用线程池处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_seq = {executor.submit(process_single_sample, task): task[1] for task in tasks}

        # 处理完成的任务
        completed = 0
        total = len(seqs)

        for future in as_completed(future_to_seq):
            seq = future_to_seq[future]
            seq_result, result = future.result()
            if result is not None:
                results[seq_result] = result
            else:
                print(f"样本 {seq} 处理失败")

            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"进度: {completed}/{total} ({completed/total*100:.1f}%)")

    return results

# 使用示例
def main(test_seqs = [22593,77736,80140,81226,91251,93219,94241,94398,94580,95689,97853,98449,100670,100801,101100,30597,30749,30972,33618,35717,38535,40984,44237,48476,50434,57735,58984,59125,61786,61942],out_file='llm_features.json'):
    # 加载训练好的模型和向量化器
    drebin_model = joblib.load('drebin_model.pkl')
    feature_vectorizer = joblib.load('tfidf_vectorizer.pkl')

    # 获取数据库连接
    conn = get_connection()

    # 提取特征
    results = extract_features_for_llm(test_seqs, drebin_model, feature_vectorizer, conn)

    # 保存结果
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


    conn.close()

if __name__ == "__main__":
    main([96760])