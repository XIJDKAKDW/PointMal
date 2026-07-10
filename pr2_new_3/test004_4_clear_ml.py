# coding=utf-8
#5、模板构建和llm调用
import os
import json
import logging
import platform
import sys
from typing import Dict, List
import pandas as pd
from datetime import datetime
from sklearn.metrics import balanced_accuracy_score

from pr2_new_3.test003 import TwoPhaseReasoningEngine, PointMalPhase3Executor

system = platform.system()
def tokenizer_func(x):
    """Tokenizer function for TF-IDF vectorizer"""
    return x.split('\n')

if system == "Linux":
    sys.path.append(r"/home/changxiaosong/python/malwareTest")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr2")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/combine_compare_tool_method")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/ganerate_pic_graph")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/AV_Agent_reimpl/test002")

from test001Method_new_9_4_3 import llm_chat

from combine_compare_tool_method import get_connection

def get_label_loop(llm_ret):
    label=''
    for one in llm_ret.split('\n'):
        if 'Final Classification' in one:
            label=one
            break
    return 0 if 'Benign' in label else 1
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
def contains(context,analysi) -> bool:
    """检查字符串是否包含指定的分析指南内容"""
    for part in analysi:
        if part.lower() not in context.lower():
            return False
    return True

class PointMalSystem:
    """
    PointMal完整系统 - 整合所有模块实现论文描述的完整流程
    """

    def __init__(self, model_paths: Dict,av_agent_output,llm_name,tempreture=None,length=None):
        self.model_paths = model_paths
        self.performance_metrics = {}
        self.interpretability_records = {}
        self.llm_name=llm_name
        self.av_agent_output=av_agent_output
        self.tempreture=tempreture
        self.length=length
    def run_complete_av_agent(self, seqs: List[int], conn, output_dir) -> Dict:
        """
        运行完整的PointMal系统
        对应论文中的完整工作流程
        """
        logger.info("开始运行完整PointMal系统")

        os.makedirs(output_dir, exist_ok=True)

        results = {}
        performance_data = []
        template_dir=r'.'+os.sep+self.av_agent_output
        for seq in seqs:
            seq=str(seq)
            logger.info(f"处理序列 {seq}")
            # 执行完整的PointMal分析流程
            template=template_dir+os.sep+f'seq_{seq}_template.json'
            with open(template, 'r', encoding='utf-8') as f:
                template = json.load(f)
            features=''
            i=0
            for key in template['key_features']:
                i+=1
                features+=(f"{i}. {key} (Importance: {template['key_features'][key]:.4f})\n")
            task = f"""
As a professional malware analysis expert, please make a judgment based on APK xml configuration, code, and the features.
# APK xml configuration:{template['theme']}
# Key Feature:[{str(features)}]
# Code: [{template['context']}]
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
            # file_path='example4.txt'
            # with open(file_path, 'r', encoding='utf-8') as f:
            #     content = f.read()
            # lsi=content.split('\n\n')
            # lsi_new=[]
            # lsi_new.extend(lsi[0].split('Output:\n'))
            # lsi_new.extend(lsi[1].split('Output:\n'))
            talks_2=[]
            # talks_2.append( {"role": "user", "content": lsi_new[0]})
            # talks_2.append({"role": "assistant", "content": lsi_new[1]})
            # talks_2.append( {"role": "user", "content": lsi_new[2]})
            # talks_2.append({"role": "assistant", "content": lsi_new[3]})
            #llm存在一定的随机性，若原样输出，则重新调用
            for i in range(5):
                talks_2, llm_ret = llm_chat(seq, talks_2.copy(), task, self.llm_name,self.tempreture,self.length)
                if 'Final Classification: [Benign/Malicious]' not in llm_ret:
                    break
            label_llm_5 = get_label_loop(llm_ret)
            reasoning_engine = TwoPhaseReasoningEngine(self.llm_name)
            av_agent_result=reasoning_engine._parse_phase2_response(llm_ret)
            av_agent_result['seq']=seq
            av_agent_result['key_features']=template['key_features']
            av_agent_result['template']=task

            av_agent_result['phase2_reasoning']=llm_ret
            av_agent_result['final_classification']='Malicious' if label_llm_5==1 else 'Benign'
            av_agent_result['confidence_scores']=template['prob']
            av_agent_result['preliminary_judgment']='Malicious' if float(template['prob'])>=0.5 else 'Benign'

            if av_agent_result:
                results[seq] = av_agent_result

                # 收集性能数据
                performance_entry = self._collect_performance_metrics(seq, av_agent_result, conn)
                performance_data.append(performance_entry)

                # 保存可解释性记录
                self._save_interpretability_record(seq, av_agent_result, output_dir)

                # 打印实时结果
                self._print_realtime_result(seq, av_agent_result)


        # 生成性能报告
        if performance_data:
            self._generate_performance_report(performance_data, output_dir)

        # 保存最终结果
        if results:
            self._save_final_results(results, output_dir)

        logger.info("完整PointMal系统执行完成")
        return results

    def _execute_av_agent_pipeline(self, seq: int, conn) -> Dict:
        """
        执行PointMal完整流水线
        对应论文中的完整算法流程
        """
        # 初始化执行器
        executor = PointMalPhase3Executor(self.model_paths)

        # 执行分析
        result = executor.execute_av_agent_analysis(seq, conn)

        # 添加时间戳和元数据
        result['timestamp'] = datetime.now().isoformat()
        result['av_agent_version'] = '1.0'
        result['pipeline_steps'] = [
        ]

        return result


    def _collect_performance_metrics(self, seq: int, result: Dict, conn) -> Dict:
        """收集性能指标"""
        # 获取真实标签
        true_label = self._get_true_label(seq, conn)

        # 获取预测结果
        pred_classification = result.get('final_classification', 'Unknown')
        pred_label = 1 if pred_classification == 'Malicious' else 0 if pred_classification == 'Benign' else -1

        # 计算置信度统计
        confidence_scores = result.get('confidence_scores', {})

        # 推理质量评估
        reasoning_quality = self._evaluate_reasoning_quality(result)

        performance_entry = {
            'seq': seq,
            'true_label': true_label,
            'predicted_label': pred_label,
            'final_classification': pred_classification,
            'avg_confidence': confidence_scores,
            'reasoning_quality': reasoning_quality,
            'timestamp': datetime.now().isoformat()
        }

        return performance_entry


    def _evaluate_reasoning_quality(self, result: Dict) -> Dict:
        """极简版推理质量评估"""
        # 默认评估结果
        quality = {
            'consistency': 'Low',
            'reasoning_depth': 'Low',
            'evidence_utilization': 'Low',
            'overall_quality': 'Low'
        }

        # 尝试获取关键信息
        final_class = result.get('final_classification', '')
        phase2_reasoning = result.get('phase2_reasoning', '')

        # 极简评估逻辑
        if final_class in ['Benign', 'Malicious']:
            quality['consistency'] = 'Medium'

        if phase2_reasoning and len(phase2_reasoning) > 50:
            quality['reasoning_depth'] = 'Medium'

        # 如果有任何可用的特征信息
        if result.get('key_features'):
            quality['evidence_utilization'] = 'Medium'

        # 极简总体质量判断
        if (quality['consistency'] == 'Medium' and
                quality['reasoning_depth'] == 'Medium'):
            quality['overall_quality'] = 'Medium'

        return quality

    def _get_true_label(self, seq: int, conn) -> int:
        """获取真实标签"""
        try:
            with conn.cursor() as cursor:
                sql = "SELECT label FROM app_label WHERE seq = %s"
                cursor.execute(sql, (seq,))
                result = cursor.fetchone()
                if result:
                    return 0 if result[0] == 'B' else 1
        except Exception as e:
            logger.error(f"获取真实标签失败: {e}")
        return -1

    def _save_interpretability_record(self, seq: int, result: Dict, output_dir: str):
        """保存可解释性记录"""
        try:
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

        except Exception as e:
            logger.error(f"可解释性记录保存失败: {e}")

    def _generate_performance_report(self, performance_data: List[Dict], output_dir: str):
        """生成性能报告"""
        df = pd.DataFrame(performance_data)

        # 计算基本指标
        total_samples = len(df)
        correct_predictions = len(df[df['true_label'] == df['predicted_label']])
        # accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        y_true = df['true_label']
        y_pred = df['predicted_label']
        # 使用平衡准确率代替普通准确率
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

        # 恶意样本检测指标
        malicious_samples = df[df['true_label'] == 1]
        true_positives = len(malicious_samples[malicious_samples['predicted_label'] == 1])
        false_negatives = len(malicious_samples[malicious_samples['predicted_label'] == 0])

        recall = true_positives / len(malicious_samples) if len(malicious_samples) > 0 else 0

        # 良性样本检测指标
        benign_samples = df[df['true_label'] == 0]
        true_negatives = len(benign_samples[benign_samples['predicted_label'] == 0])
        false_positives = len(benign_samples[benign_samples['predicted_label'] == 1])

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

        # 推理质量统计
        reasoning_qualities = [entry['reasoning_quality']['overall_quality'] for entry in performance_data]
        quality_distribution = {
            'High': reasoning_qualities.count('High'),
            'Medium': reasoning_qualities.count('Medium'),
            'Low': reasoning_qualities.count('Low')
        }

        # 生成报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': total_samples,
            'balanced_accuracy': balanced_accuracy,
            'recall': recall,
            'precision': precision,
            'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
            'avg_confidence': df['avg_confidence'].astype(float).mean(),
            'reasoning_quality_distribution': quality_distribution,
            'detailed_metrics': {
                'true_positives': true_positives,
                'false_positives': false_positives,
                'true_negatives': true_negatives,
                'false_negatives': false_negatives
            }
        }

        # 保存报告
        report_file = os.path.join(output_dir, "acc_f1_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 生成CSV格式的详细数据
        csv_file = os.path.join(output_dir, "acc_f1_details.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8')

        logger.info(f"性能报告生成完成: {report_file}")

    def _save_final_results(self, results: Dict, output_dir: str):
        """保存最终结果"""
        # 保存完整结果
        final_file = os.path.join(output_dir, "av_agent_final_results.json")
        with open(final_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        # 生成摘要报告
        summary = self._generate_summary_report(results)
        summary_file = os.path.join(output_dir, "summary_report.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"最终结果保存完成: {final_file}")


    def _generate_summary_report(self, results: Dict) -> Dict:
        """生成摘要报告"""
        summary = {
            'total_processed': len(results),
            'classification_distribution': {},
            'avg_confidence_by_type': {},
            'reasoning_quality_summary': {}
        }

        # 分类分布
        classifications = [result.get('final_classification', 'Unknown') for result in results.values()]
        summary['classification_distribution'] = {
            'Malicious': classifications.count('Malicious'),
            'Benign': classifications.count('Benign'),
            'Unknown': classifications.count('Unknown')
        }


        return summary

    def _print_realtime_result(self, seq: int, result: Dict):
        """打印实时结果"""
        final_classification = result.get('final_classification', 'Unknown')
        phase1_judgment = result.get('preliminary_judgment', 'Unknown')
        phase2_confidence = result.get('final_confidence', 'Unknown')

        print(f"[PointMal] Sequence {seq}: Phase1 Judgment={phase1_judgment}, "
              f"Final Classification={final_classification}, Confidence={phase2_confidence}")

def main(test_seqs, PointMal_output, av_agent_final, llm_name, t=None, leng=None):
    """全部（特征+xml+函数）"""
    conn = get_connection()
    if conn is None:
        logger.error("无法连接到数据库")
        return

    # 模型路径配置
    model_paths = {
        'drebin': 'drebin_model.pkl',
    }

    # 测试序列

    try:
        # 初始化PointMal系统
        av_agent_system = PointMalSystem(model_paths, PointMal_output, llm_name, tempreture=t, length=leng)

        # 运行完整系统
        results = av_agent_system.run_complete_av_agent(test_seqs, conn,av_agent_final)

        print(f"\n=== PointMal系统执行完成 ===")
        print(f"处理样本数: {len(results)}")

        # 打印最终统计
        malicious_count = sum(1 for r in results.values() if r.get('final_classification') == 'Malicious')
        benign_count = sum(1 for r in results.values() if r.get('final_classification') == 'Benign')

        print(f"恶意分类: {malicious_count}")
        print(f"良性分类: {benign_count}")
        print(f"详细报告请查看输出目录")

    finally:
        conn.close()
        logger.info("数据库连接已关闭")

if __name__ == "__main__":
    av_agent_output,llm_name='av_agent_output','codellama:7b'
    av_agent_final='av_agent_final_1'
    main([102135,102593,105458,105541,105674,107027,107183,107227,107492,109539,109820,109927,110407,111034,11144,112153,112235,112246,112690,113381,113625,113700,113753,113761,114044,114812,115140,116513,117695,128864,129113,13643,136458,138414,13997,14371,144695,145349,147590,15564,160751,16504,16637,21195,24940,25362,26282,26498,35347,36188,36962,52436,55149,56260,63280,7043,72568,78154,78190,89738,90919,93865,97048,98609],av_agent_output,av_agent_final,llm_name)