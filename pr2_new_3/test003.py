# coding=utf-8
import json
import logging
import os
import platform
import sys
from typing import Dict, List

import requests

from pr2_new_3.test001Method_new_9_4_3 import get_xml, llm_chat

system = platform.system()

if system == "Linux":
    sys.path.append(r"/home/changxiaosong/python/malwareTest")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr2")




def tokenizer_func(x):
    """Tokenizer function for TF-IDF vectorizer"""
    return x.split('\n')

from combine_compare_tool_method import get_connection

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class TwoPhaseReasoningEngine:
    """
    两阶段推理引擎 - 实现论文中的两阶段推理机制
    对应论文IV-C节: Two-Phase Reasoning
    """

    def __init__(self, llm_name):
        self.llm_name=llm_name
    def _parse_phase2_response(self, response: str) -> Dict:
        """Parse phase 2 response"""
        parsed = {
            'final_classification': 'Unknown',
            'key_evidence': '',
            'behavior_pattern': '',
            'detailed_reasoning': '',
            'final_confidence': 'Low'
        }

        try:
            lines = response.split('\n')
            current_section = None

            for line in lines:
                line = line.strip()
                if line.startswith('Final Classification:'):
                    parsed['final_classification'] = line.replace('Final Classification:', '').strip()
                    current_section = None
                elif line.startswith('Key Evidence:'):
                    current_section = 'key_evidence'
                    parsed['key_evidence'] = line.replace('Key Evidence:', '').strip()
                elif line.startswith('Behavior Pattern:'):
                    current_section = 'behavior_pattern'
                    parsed['behavior_pattern'] = line.replace('Behavior Pattern:', '').strip()
                elif line.startswith('Detailed Reasoning:'):
                    current_section = 'detailed_reasoning'
                    parsed['detailed_reasoning'] = line.replace('Detailed Reasoning:', '').strip()
                elif line.startswith('Final Confidence:'):
                    parsed['final_confidence'] = line.replace('Final Confidence:', '').strip()
                    current_section = None
                elif current_section and line:
                    # Append content to current section
                    parsed[current_section] += ' ' + line

        except Exception as e:
            logger.warning(f"Failed to parse phase 2 response: {e}")

        return parsed

    def get_context(self, RM, CM, AC, AI, key_features: Dict) -> Dict:
        """
        第一阶段推理 - 基于置信度分数的分析
        对应论文中的 R1_LLM : T ! R1
        """
        logger.info("得到背景")
        return RM

    def get_theme(self, seq) -> Dict:
        logger.info("主题抽取")
        smali_tmp=r'..\pr2\decompiled_java\smali_tmp'
        if system == "Linux":
            smali_tmp='/home/changxiaosong/python/malwareTest/pr2'+os.sep+'decompiled_java'+os.sep+'smali_tmp'
        xml=get_xml(seq,smali_tmp)
        return xml

        prompt = f"""Guess the functionality of this APK based on the configuration file content.
        Output strictly in the following format, with no other content:
        This APK is a [application_type] application that includes [feature1], [feature2], and [feature3]."""
        talks = []
        talks.append( {"role": "user", "content": f'''## Configuration file content:{xml}'''})
        _, tmp = llm_chat('', talks, prompt, self.llm_name)
        return tmp



    def _format_confidence_scores(self, confidence_scores: Dict) -> str:
        """Format confidence scores"""
        if not confidence_scores:
            return "No valid confidence data"

        formatted = ""
        for feature_type, scores in confidence_scores.items():
            confidence = scores.get('confidence', 0)
            predicted_class = scores.get('predicted_class', 0)
            description = scores.get('description', feature_type)

            risk_level = "High risk" if confidence > 0.7 else "Medium risk" if confidence > 0.5 else "Low risk"
            pred_label = "Malicious" if predicted_class == 1 else "Benign"

            formatted += f"- {description}: {confidence:.4f} ({risk_level}, Prediction: {pred_label})\n"

        # 添加统计摘要
        if confidence_scores:
            conf_values = [scores.get('confidence', 0) for scores in confidence_scores.values()]
            avg_confidence = sum(conf_values) / len(conf_values)
            max_confidence = max(conf_values)

            formatted += f"\nStatistical Summary:\n"
            formatted += f"- Average Confidence: {avg_confidence:.4f}\n"
            formatted += f"- Highest Confidence: {max_confidence:.4f}\n"
            formatted += f"- Number of Classifiers: {len(confidence_scores)}\n"
        return formatted

    def _format_key_features(self, key_features: Dict) -> Dict:
        feature = []
        i=0
        for one in key_features:
            i+=1
            feature.append( f"{i}. {one['feature_name']} (Importance: {one['shap_value']:.4f})")
        return feature

    def get_key_feature_name(self, key_features: Dict) -> Dict:
        feature = []
        i=0
        for one in key_features:
            i+=1
            feature.append(one['feature_name'])
        return feature

    def _extract_raw_bytes_info(self, raw_bytes_data: Dict) -> str:
        """从raw_bytes数据中提取关键信息"""
        try:
            info_parts = []

            # 文件头信息
            if raw_bytes_data.get('file_header'):
                header = raw_bytes_data['file_header']
                info_parts.append(f"File Header: {header}")

            # 节区信息
            if raw_bytes_data.get('sections'):
                sections = raw_bytes_data['sections']
                info_parts.append(f"Sections: {len(sections)} sections detected")
                for i, section in enumerate(sections[:3]):  # 只显示前3个节区
                    info_parts.append(f"  - Section {i+1}: {section.get('name', 'Unknown')}")

            # 导入表信息
            if raw_bytes_data.get('imports'):
                imports = raw_bytes_data['imports']
                info_parts.append(f"Imports: {len(imports)} imported functions")
                suspicious_imports = [imp for imp in imports if self._is_suspicious_import(imp)]
                if suspicious_imports:
                    info_parts.append("Suspicious Imports:")
                    for imp in suspicious_imports[:5]:
                        info_parts.append(f"  - {imp}")

            # 字符串特征
            if raw_bytes_data.get('strings'):
                strings = raw_bytes_data['strings']
                suspicious_strings = [s for s in strings if self._is_suspicious_string(s)]
                if suspicious_strings:
                    info_parts.append(f"Suspicious Strings: {len(suspicious_strings)} found")
                    for s in suspicious_strings[:5]:
                        info_parts.append(f"  - {s}")

            return "\n".join(info_parts) if info_parts else "Limited raw bytes information available"

        except Exception as e:
            logger.warning(f"Error extracting raw bytes info: {e}")
            return "Raw bytes information processing failed"


    def _is_suspicious_import(self, import_name: str) -> bool:
        """判断导入函数是否可疑"""
        suspicious_keywords = [
            'crypt', 'encrypt', 'decrypt', 'virtualalloc', 'virtualprotect',
            'createremotethread', 'setwindowshook', 'regsetvalue',
            'getkeystate', 'shellexecute', 'winexec'
        ]
        import_lower = import_name.lower()
        return any(keyword in import_lower for keyword in suspicious_keywords)

    def _is_suspicious_string(self, string: str) -> bool:
        """判断字符串是否可疑"""
        suspicious_patterns = [
            'http://', 'https://', '.exe', '.dll', 'registry',
            'autostart', 'startup', 'password', 'keylogger'
        ]
        string_lower = string.lower()
        return any(pattern in string_lower for pattern in suspicious_patterns)


    def _parse_phase1_response(self, response: str) -> Dict:
        """Parse phase 1 response"""
        parsed = {
            'preliminary_judgment': 'Unknown',
            'main_evidence': '',
            'confidence_assessment': 'Low',
            'reasoning_summary': ''
        }

        try:
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('Preliminary Judgment:'):
                    parsed['preliminary_judgment'] = line.replace('Preliminary Judgment:', '').strip()
                elif line.startswith('Main Evidence:'):
                    parsed['main_evidence'] = line.replace('Main Evidence:', '').strip()
                elif line.startswith('Confidence Assessment:'):
                    parsed['confidence_assessment'] = line.replace('Confidence Assessment:', '').strip()
                elif line.startswith('Reasoning Summary:'):
                    parsed['reasoning_summary'] = line.replace('Reasoning Summary:', '').strip()

        except Exception as e:
            logger.warning(f"Failed to parse phase 1 response: {e}")

        return parsed

    def _parse_phase2_response(self, response: str) -> Dict:
        """Parse phase 2 response"""
        parsed = {
            'final_classification': 'Unknown',
            'key_evidence': '',
            'behavior_pattern': '',
            'detailed_reasoning': '',
            'final_confidence': 'Low'
        }

        try:
            lines = response.split('\n')
            current_section = None

            for line in lines:
                line = line.strip()
                if line.startswith('Final Classification:'):
                    parsed['final_classification'] = line.replace('Final Classification:', '').strip()
                    current_section = None
                elif line.startswith('Key Evidence:'):
                    current_section = 'key_evidence'
                    parsed['key_evidence'] = line.replace('Key Evidence:', '').strip()
                elif line.startswith('Behavior Pattern:'):
                    current_section = 'behavior_pattern'
                    parsed['behavior_pattern'] = line.replace('Behavior Pattern:', '').strip()
                elif line.startswith('Detailed Reasoning:'):
                    current_section = 'detailed_reasoning'
                    parsed['detailed_reasoning'] = line.replace('Detailed Reasoning:', '').strip()
                elif line.startswith('Final Confidence:'):
                    parsed['final_confidence'] = line.replace('Final Confidence:', '').strip()
                    current_section = None
                elif current_section and line:
                    # Append content to current section
                    parsed[current_section] += ' ' + line

        except Exception as e:
            logger.warning(f"Failed to parse phase 2 response: {e}")

        return parsed

class PointMalPhase3Executor:
    """
    PointMal第三阶段执行器 - 整合特征工程和两阶段推理
    """

    def __init__(self, model_paths: Dict,llm_name,out_file):
        self.model_paths = model_paths
        self.llm_name=llm_name
        with open(out_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)


    def execute_av_agent_analysis(self, seq: int, llm_name) -> Dict:
        reasoning_engine = TwoPhaseReasoningEngine(llm_name)
        final_result={}
        if seq in self.data:
            RM,CM,AC,AI=self.data[seq]['RM_str'],self.data[seq]['CM_str'],self.data[seq]['AC_str'],self.data[seq]['AI_str']
            key_features={}
            for one in self.data[seq]['key_features']:
                if len(one['feature_name'])>0:
                    key_features[one['feature_name']]=one['shap_value']
            theme=reasoning_engine.get_theme(seq)
            context=reasoning_engine.get_context(RM,CM,AC,AI,self.data[seq]['key_features'])
            prob=self.data[seq]['confidence_scores']['drebin']['confidence']
            final_result = {
                'seq': seq,
                'theme': theme,
                'RM': RM,
                'CM': CM,
                'AC': AC,
                'AI': AI,
                'context': context,
                'prob': prob,
                'key_features': key_features
            }
        logger.info(f"序列 {seq} PointMal分析完成")
        return final_result


def run_av_agent_analysis(seqs: List[int], conn, output_dir,llm_name,out_file='llm_features.json'):
    """运行PointMal分析"""
    logger.info("开始PointMal分析流程")

    os.makedirs(output_dir, exist_ok=True)

    # 模型路径配置
    model_paths = {
        'raw_bytes': 'raw_bytes_model.pth',
        'drebin': 'drebin_model.pkl',
        'image': 'image_model.pth'
    }

    # 初始化执行器
    executor = PointMalPhase3Executor(model_paths, llm_name, out_file)

    results = {}
    for seq in seqs:
        seq=str(seq)
        output_file = os.path.join(output_dir, f"seq_{seq}_template.json")
        logger.info(f"模板生成，seq： {seq}")
        # 执行PointMal分析
        result = executor.execute_av_agent_analysis(seq, llm_name)
        results[seq] = result

        # 保存单个结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        # if not os.path.exists(output_file):
        #     pass

    return results

def main(test_seqs,output_dir,llm_name,out_file='llm_features.json'):
    """PointMal使用示例"""
    conn = get_connection()
    if conn is None:
        logger.error("无法连接到数据库")
        return


    try:
        results = run_av_agent_analysis(test_seqs, conn,output_dir,llm_name,out_file)
    finally:
        conn.close()
        logger.info("数据库连接已关闭")

if __name__ == "__main__":
    output_dir,llm_name='av_agent_output','deepseek-chat'
    main([102135,102593,105458,105541,105674,107027,107183,107227,107492,109539,109820,109927,110407,111034,11144,112153,112235,112246,112690,113381,113625,113700,113753,113761,114044,114812,115140,116513,117695,128864,129113,13643,136458,138414,13997,14371,144695,145349,147590,15564,160751,16504,16637,21195,24940,25362,26282,26498,35347,36188,36962,52436,55149,56260,63280,7043,72568,78154,78190,89738,90919,93865,97048,98609],output_dir,llm_name)