import json
import os
import platform
import random
import sys

import pandas as pd


system = platform.system()

if system == "Linux":
    sys.path.append(r"/home/changxiaosong/python/malwareTest")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr2_new_3")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr2_new_2")

from pr2_new_3 import test003, test001, test002, test004_2, test004_3, test004_1, test004_4
from pr2_new_2 import GetMultipleMetrixMethod_2

from test001Method_new_9_4_3 import plot_value

def get_pre_result_one(folder):
    file_path = os.path.join(folder, 'acc_f1_report.json')
    with open(file_path, 'r') as f:
        result = json.load(f)
    print(['llm-result ',folder,result['balanced_accuracy'], result['precision'], result['recall'], result['f1_score']])

def get_pre_result(f1, f2, f3):
    # 读取所有文件
    dfs = []
    folders = [f1, f2, f3]
    for i, folder in enumerate(folders):
        file_path = os.path.join(folder, 'acc_f1_details.csv')
        df = pd.read_csv(file_path)
        df = df[['seq', 'true_label', 'predicted_label']]
        df = df.rename(columns={'predicted_label': f'pred_{i+1}'})
        if i == 0:
            df = df.rename(columns={'true_label': 'true_label'})
        else:
            df = df.drop(columns=['true_label'])
        dfs.append(df)

    # 按seq合并所有DataFrame
    merged_df = dfs[0]
    for i in range(1, len(dfs)):
        merged_df = pd.merge(merged_df, dfs[i], on='seq', how='inner')

    # 确保数据按seq排序
    merged_df = merged_df.sort_values('seq').reset_index(drop=True)

    true_labels = merged_df['true_label'].tolist()
    pred_columns = [f'pred_{i+1}' for i in range(len(folders))]

    # 投票逻辑
    final_predictions = []
    for _, row in merged_df.iterrows():
        votes = [row[col] for col in pred_columns]
        positive_votes = sum(votes)
        final_predictions.append(1 if positive_votes >= 3 else 0)
    balance_accuracy_llm, precision_llm, recall_llm, f1_value_llm = \
        plot_value(true_labels, final_predictions)
    for i, folder in enumerate(folders):
        file_path = os.path.join(folder, 'acc_f1_report.json')
        with open(file_path, 'r') as f:
            result = json.load(f)
        print(['llm-result ',folder,result['balanced_accuracy'], result['precision'], result['recall'], result['f1_score']])
    print(['llm-result 三类投票：',balance_accuracy_llm, precision_llm, recall_llm, f1_value_llm])
import os
import pandas as pd
import glob

def get_final_csv(ml_pre_dict, PointMal_final_1, PointMal_final_2, PointMal_final_3):
    # 加载三个CSV文件
    csv_files = []
    agent_names = []

    for i, agent_path in enumerate([PointMal_final_1, PointMal_final_2, PointMal_final_3]):
        csv_path = os.path.join(agent_path, 'acc_f1_details.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            agent_name = f'llm_label_av_agent_final_{i+1}'
            agent_names.append(agent_name)

            df_selected = df[['seq', 'true_label', 'final_classification']].copy()
            df_selected.rename(columns={'final_classification': agent_name}, inplace=True)
            csv_files.append(df_selected)
        else:
            print(f"Warning: {csv_path} not found")
    # 合并三个CSV文件（使用outer join确保包含所有seq）
    combined_df = csv_files[0]
    for df in csv_files[1:]:
        combined_df = pd.merge(combined_df, df, on=['seq', 'true_label'], how='outer')

    # 添加ML标签
    ml_labels = []
    for seq in combined_df['seq']:
        seq_str = str(seq)
        seq_int = int(seq)
        if seq_int in ml_pre_dict:
            # 根据概率阈值0.5确定ML标签
            ml_prob = float(ml_pre_dict[seq_int]) if isinstance(ml_pre_dict[seq_int], str) else ml_pre_dict[seq_int]
            ml_labels.append(1 if ml_prob >= 0.5 else 0)
        else:
            ml_labels.append(None)

    combined_df['ml_label'] = ml_labels

    # 重新排列列顺序
    columns_order = ['seq', 'true_label', 'ml_label'] + agent_names
    combined_df = combined_df[columns_order]

    # 保存合并后的CSV
    output_path = 'acc_f1_details_comb.csv'
    combined_df.to_csv(output_path, index=False)
    print(f"Combined CSV saved to {output_path}")

    # 打印统计信息
    print(f"\n合并后的数据统计:")
    print(f"总样本数: {len(combined_df)}")



if __name__ == '__main__':
    train_file=r'/home/changxiaosong/python/malwareTest/train_0.5begin-dowgin-smsreg-smssend-.txt'
    test_file=r'/home/changxiaosong/python/malwareTest/test_0.5begin-dowgin-smsreg-smssend-.txt'
    output_dir='av_agent_output'

    llm_name='deepseek-coder-v2:16b'
    av_agent_final_1='av_agent_final_1'
    av_agent_final_2='av_agent_final_2'
    av_agent_final_3='av_agent_final_3'
    av_agent_final_4='av_agent_final_4'


    seqs_test = test001.load_seqs_from_file(test_file)
    # seqs_test=[64854,47853,130737,116979,148366,101183,91725,105827,109315,10903]#random.sample(seqs_test, 10)

    # ml_pre_dict=test001.main(train_file,seqs_test)#模型训练
    # test002.main(seqs_test)#可疑代码提取
    # test003.main(seqs_test,output_dir,llm_name)#模板生成
    # #调用llm推理
    # xml
    # test004_1.main(seqs_test,output_dir,av_agent_final_1,llm_name)
    # #+risky method
    # test004_2.main(seqs_test,output_dir,av_agent_final_2,llm_name)
    # #+特征
    # test004_3.main(seqs_test,output_dir,av_agent_final_3,llm_name)
    #全部
    test004_4.main(seqs_test,output_dir,av_agent_final_4,llm_name)


    # get_pre_result(av_agent_final_1,av_agent_final_2,av_agent_final_3)
    get_pre_result_one(av_agent_final_4)
    # get_final_csv(ml_pre_dict,av_agent_final_1,av_agent_final_2,av_agent_final_3)

