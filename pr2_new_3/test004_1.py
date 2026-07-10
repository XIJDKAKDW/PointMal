# coding=utf-8
#5、模板构建和llm调用 - 多线程版本
import os
import json
import logging
import platform
import sys
from typing import Dict, List
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import balanced_accuracy_score

from pr2_new_3.test003 import TwoPhaseReasoningEngine, PointMalPhase3Executor

system = platform.system()

if system == "Linux":
    sys.path.append(r"/home/changxiaosong/python/malwareTest")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr2")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/combine_compare_tool_method")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/ganerate_pic_graph")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/AV_Agent_reimpl/test002")

from test001Method_new_9_4_3 import llm_chat
from combine_compare_tool_method import get_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def get_label_loop(llm_ret):
    label = ''
    for one in llm_ret.split('\n'):
        if 'Final Classification' in one:
            label = one
            break
    return 0 if 'Benign' in label else 1

class AVAgentSystem:
    def __init__(self, model_paths: Dict, av_agent_output, llm_name, tempreture=None, length=None, max_workers=4):
        self.model_paths = model_paths
        self.performance_metrics = {}
        self.interpretability_records = {}
        self.llm_name = llm_name
        self.av_agent_output = av_agent_output
        self.tempreture = tempreture
        self.length = length
        self.max_workers = max_workers  # 线程池大小

    def _process_single_seq(self, seq: str) -> Dict:
        """处理单个序列，供线程调用"""
        try:
            logger.info(f"处理序列 {seq}")
            template_dir = r'.' + os.sep + self.av_agent_output
            template_path = template_dir + os.sep + f'seq_{seq}_template.json'

            with open(template_path, 'r', encoding='utf-8') as f:
                template = json.load(f)

            if template is None or template.get('prob') is None:
                return None

            features = ''
            i = 0
            for key in template['key_features']:
                i += 1
                features += (f"{i}. {key} (Importance: {template['key_features'][key]:.4f})\n")

            task = f"""
As a professional malware analysis expert, please make a judgment based on the preliminary detection result, and APK xml configuration.
# Initial machine learning detection result: [{'' if template['prob'] is None else 'Malicious' if float(template['prob'])>=0.5 else 'Benign'}] 
# Classifier Confidence Score [ {'' if template['prob'] is None else template['prob']}]
# APK xml configuration:{template['theme']}
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
            talks_2 = []
            llm_ret = None
            for i in range(5):
                talks_2, llm_ret = llm_chat(seq, talks_2.copy(), task, self.llm_name, self.tempreture, self.length)
                if 'Final Classification: [Benign/Malicious]' not in llm_ret:
                    break

            label_llm_5 = get_label_loop(llm_ret)
            reasoning_engine = TwoPhaseReasoningEngine(self.llm_name)
            av_agent_result = reasoning_engine._parse_phase2_response(llm_ret)
            av_agent_result['seq'] = seq
            av_agent_result['key_features'] = template['key_features']
            av_agent_result['template'] = task
            av_agent_result['phase2_reasoning'] = llm_ret
            av_agent_result['final_classification'] = 'Malicious' if label_llm_5 == 1 else 'Benign'
            av_agent_result['confidence_scores'] = template['prob']
            av_agent_result['preliminary_judgment'] = 'Malicious' if float(template['prob']) >= 0.5 else 'Benign'

            return av_agent_result

        except Exception as e:
            logger.error(f"处理序列 {seq} 失败: {e}")
            return None

    def run_complete_av_agent(self, seqs: List[int], output_dir) -> Dict:
        """多线程并行运行完整的PointMal系统"""
        logger.info("开始运行完整PointMal系统（多线程模式）")
        os.makedirs(output_dir, exist_ok=True)

        results = {}
        performance_data = []

        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_seq = {
                executor.submit(self._process_single_seq, str(seq)): seq
                for seq in seqs
            }

            for future in as_completed(future_to_seq):
                seq = future_to_seq[future]
                try:
                    av_agent_result = future.result(timeout=300)
                    if av_agent_result:
                        results[str(seq)] = av_agent_result
                        performance_entry = self._collect_performance_metrics(str(seq), av_agent_result)
                        performance_data.append(performance_entry)
                        self._save_interpretability_record(str(seq), av_agent_result, output_dir)
                        self._print_realtime_result(seq, av_agent_result)
                except Exception as e:
                    logger.error(f"获取序列 {seq} 结果失败: {e}")

        # 生成报告
        if performance_data:
            self._generate_performance_report(performance_data, output_dir)
        if results:
            self._save_final_results(results, output_dir)

        logger.info("完整PointMal系统执行完成")
        return results

    # 以下方法保持不变
    def _execute_av_agent_pipeline(self, seq: int, conn) -> Dict:
        executor = PointMalPhase3Executor(self.model_paths)
        result = executor.execute_av_agent_analysis(seq, conn)
        result['timestamp'] = datetime.now().isoformat()
        result['av_agent_version'] = '1.0'
        result['pipeline_steps'] = []
        return result

    def _collect_performance_metrics(self, seq: str, result: Dict) -> Dict:
        conn = get_connection()
        try:
            true_label = self._get_true_label(int(seq), conn)
            pred_classification = result.get('final_classification', 'Unknown')
            pred_label = 1 if pred_classification == 'Malicious' else 0 if pred_classification == 'Benign' else -1
            confidence_scores = result.get('confidence_scores', {})
            reasoning_quality = self._evaluate_reasoning_quality(result)
        finally:
            conn.close()
        return {
            'seq': seq,
            'true_label': true_label,
            'predicted_label': pred_label,
            'final_classification': pred_classification,
            'avg_confidence': confidence_scores,
            'reasoning_quality': reasoning_quality,
            'timestamp': datetime.now().isoformat()
        }

    def _evaluate_reasoning_quality(self, result: Dict) -> Dict:
        quality = {'consistency': 'Low', 'reasoning_depth': 'Low', 'evidence_utilization': 'Low', 'overall_quality': 'Low'}
        final_class = result.get('final_classification', '')
        phase2_reasoning = result.get('phase2_reasoning', '')

        if final_class in ['Benign', 'Malicious']:
            quality['consistency'] = 'Medium'
        if phase2_reasoning and len(phase2_reasoning) > 50:
            quality['reasoning_depth'] = 'Medium'
        if result.get('key_features'):
            quality['evidence_utilization'] = 'Medium'
        if quality['consistency'] == 'Medium' and quality['reasoning_depth'] == 'Medium':
            quality['overall_quality'] = 'Medium'
        return quality

    def _get_true_label(self, seq: int, conn) -> int:
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT label FROM app_label WHERE seq = %s", (seq,))
                result = cursor.fetchone()
                if result:
                    return 0 if result[0] == 'B' else 1
        except Exception as e:
            logger.error(f"获取真实标签失败: {e}")
        return -1

    def _save_interpretability_record(self, seq: str, result: Dict, output_dir: str):
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
        df = pd.DataFrame(performance_data)
        y_true = df['true_label']
        y_pred = df['predicted_label']
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

        malicious_samples = df[df['true_label'] == 1]
        true_positives = len(malicious_samples[malicious_samples['predicted_label'] == 1])
        false_negatives = len(malicious_samples[malicious_samples['predicted_label'] == 0])
        recall = true_positives / len(malicious_samples) if len(malicious_samples) > 0 else 0

        benign_samples = df[df['true_label'] == 0]
        false_positives = len(benign_samples[benign_samples['predicted_label'] == 1])
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

        reasoning_qualities = [entry['reasoning_quality']['overall_quality'] for entry in performance_data]
        quality_distribution = {
            'High': reasoning_qualities.count('High'),
            'Medium': reasoning_qualities.count('Medium'),
            'Low': reasoning_qualities.count('Low')
        }

        report = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(df),
            'balanced_accuracy': balanced_accuracy,
            'recall': recall,
            'precision': precision,
            'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
            'avg_confidence': df['avg_confidence'].astype(float).mean(),
            'reasoning_quality_distribution': quality_distribution,
            'detailed_metrics': {'true_positives': true_positives, 'false_positives': false_positives,
                                 'true_negatives': len(benign_samples[benign_samples['predicted_label'] == 0]),
                                 'false_negatives': false_negatives}
        }

        report_file = os.path.join(output_dir, "acc_f1_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        df.to_csv(os.path.join(output_dir, "acc_f1_details.csv"), index=False, encoding='utf-8')
        logger.info(f"性能报告生成完成: {report_file}")

    def _save_final_results(self, results: Dict, output_dir: str):
        final_file = os.path.join(output_dir, "av_agent_final_results.json")
        with open(final_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        summary = self._generate_summary_report(results)
        with open(os.path.join(output_dir, "summary_report.json"), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"最终结果保存完成: {final_file}")

    def _generate_summary_report(self, results: Dict) -> Dict:
        classifications = [result.get('final_classification', 'Unknown') for result in results.values()]
        return {
            'total_processed': len(results),
            'classification_distribution': {
                'Malicious': classifications.count('Malicious'),
                'Benign': classifications.count('Benign'),
                'Unknown': classifications.count('Unknown')
            },
            'avg_confidence_by_type': {},
            'reasoning_quality_summary': {}
        }

    def _print_realtime_result(self, seq: int, result: Dict):
        print(f"[PointMal] Sequence {seq}: Phase1 Judgment={result.get('preliminary_judgment', 'Unknown')}, "
              f"Final Classification={result.get('final_classification', 'Unknown')}, "
              f"Confidence={result.get('final_confidence', 'Unknown')}")

def main(test_seqs, av_agent_output, av_agent_final, llm_name, t=None, leng=None, max_workers=4):
    #代码
    model_paths = {'drebin': 'drebin_model.pkl'}

    av_agent_system = AVAgentSystem(model_paths, av_agent_output, llm_name, tempreture=t, length=leng, max_workers=max_workers)
    results = av_agent_system.run_complete_av_agent(test_seqs, av_agent_final)

    print(f"\n=== PointMal系统执行完成 ===")
    print(f"处理样本数: {len(results)}")
    malicious_count = sum(1 for r in results.values() if r.get('final_classification') == 'Malicious')
    benign_count = sum(1 for r in results.values() if r.get('final_classification') == 'Benign')
    print(f"恶意分类: {malicious_count}")
    print(f"良性分类: {benign_count}")

if __name__ == "__main__":
    seqs = [102135,102593,105458,105541,105674,107027,107183,107227,107492,109539,109820,109927,110407,111034,11144,112153,112235,112246,112690,113381,113625,113700,113753,113761,114044,114812,115140,116513,117695,128864,129113,13643,136458,138414,13997,14371,144695,145349,147590,15564,160751,16504,16637,21195,24940,25362,26282,26498,35347,36188,36962,52436,55149,56260,63280,7043,72568,78154,78190,89738,90919,93865,97048,98609]
    main(seqs, 'av_agent_output', 'av_agent_final_1', 'codellama:7b', max_workers=4)