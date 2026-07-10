#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：malwareTest 
@File    ：exp_3.py.py
@IDE     ：PyCharm 
@Author  ：常晓松
@Date    ：2026/3/27 14:33 
'''
import argparse
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from sklearn.neural_network import MLPClassifier
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings('ignore')
import platform
import sys
system = platform.system()

if system == "Linux":
    sys.path.append(r"/home/changxiaosong/python/malwareTest")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr2_final")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr2_new_2")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr3")
from pr2_new_3 import test001, test004_1, test004_2, test004_3, test004_4, test004_4_clear_ml
from pr2_new_3.test001 import get_label_by_seq, get_connection, plot_value

from pr2_new_3.exp_3 import train_model_finetune
from pr2_new_3.Main_within import get_final_csv


def get_json_content(dir,one):
    file_path=f'seq_{one}_interpretability.json'
    inf=''
    with open(dir+os.sep+file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        inf=data['template']+'----------------'+data['phase2_reasoning']
    return inf


def prepare_data(train_data, test_data):
    """准备训练和测试数据"""
    print("准备训练数据...")
    X_train = []
    y_train = []

    for seq, data in train_data.items():
        # 组合三个通道的内容
        combined_text = f""
        combined_text += f"=== CHANNEL_1 ===\n{data.get(view_train_1, '')}\n\n"
        combined_text += f"=== CHANNEL_2 ===\n{data.get(view_train_2, '')}\n\n"
        combined_text += f"=== CHANNEL_3 ===\n{data.get(view_train_3, '')}"

        X_train.append(combined_text)
        y_train.append(data['label'])

    print("准备测试数据...")
    X_test = []
    y_test = []

    for seq, data in test_data.items():
        combined_text = f""
        combined_text += f"=== CHANNEL_1 ===\n{data.get(view_test_1, '')}\n\n"
        combined_text += f"=== CHANNEL_2 ===\n{data.get(view_test_2, '')}\n\n"
        combined_text += f"=== CHANNEL_3 ===\n{data.get(view_test_3, '')}"

        X_test.append(combined_text)
        y_test.append(data['label'])

    return X_train, y_train, X_test, y_test


def prepare_data_not_mv(train_data, test_data):
    """准备训练和测试数据"""
    print("准备训练数据...")
    X_train = []
    y_train = []

    for seq, data in train_data.items():
        # 组合三个通道的内容
        combined_text = f""
        combined_text += f"=== CHANNEL_1 ===\n{data.get(view_train_4, '')}\n\n"

        X_train.append(combined_text)
        y_train.append(data['label'])

    print("准备测试数据...")
    X_test = []
    y_test = []

    for seq, data in test_data.items():
        combined_text = f""
        combined_text += f"=== CHANNEL_1 ===\n{data.get(view_test_4, '')}\n\n"

        X_test.append(combined_text)
        y_test.append(data['label'])

    return X_train, y_train, X_test, y_test
def get_embedding_from_text(text, model, tokenizer, device):
    """获取文本嵌入向量"""
    if not text:
        return None

    inputs = tokenizer(text, return_tensors="pt", max_length=2048, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        embedding = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

    return embedding.cpu().numpy().squeeze()

def train_model(X_train_emb, y_train):
    # 训练MLP
    print("\n训练MLP分类器...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation='relu',
        max_iter=300,
        random_state=42,
        verbose=True
    )
    mlp.fit(X_train_emb, y_train)
    return mlp

def get_vec(device,X_train,  X_test):
    """训练模型"""
    # 加载DeepSeek模型
    print("加载DeepSeek模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model.to(device)
    model.eval()

    # 生成嵌入向量
    print("生成训练集嵌入向量...")
    X_train_emb = []
    for i, text in enumerate(X_train):
        emb = get_embedding_from_text(text, model, tokenizer, device)
        if emb is not None:
            X_train_emb.append(emb)
        if (i+1) % 50 == 0:
            print(f"  进度: {i+1}/{len(X_train)}")

    print("生成测试集嵌入向量...")
    X_test_emb = []
    for i, text in enumerate(X_test):
        emb = get_embedding_from_text(text, model, tokenizer, device)
        if emb is not None:
            X_test_emb.append(emb)
        if (i+1) % 50 == 0:
            print(f"  进度: {i+1}/{len(X_test)}")

    X_train_emb = np.array(X_train_emb)
    X_test_emb = np.array(X_test_emb)

    print(f"训练集维度: {X_train_emb.shape}")
    print(f"测试集维度: {X_test_emb.shape}")
    return X_train_emb,X_test_emb
def get_pre_result_one(folder):
    file_path = os.path.join(folder, 'acc_f1_report.json')
    with open(file_path, 'r') as f:
        result = json.load(f)
    return result['balanced_accuracy'], result['precision'], result['recall'], result['f1_score']

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
    return balance_accuracy_llm, precision_llm, recall_llm, f1_value_llm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True, help='训练集文件路径')
    parser.add_argument('--test', required=True, help='测试集文件路径')
    args = parser.parse_args()

    train_file = args.train
    test_file = args.test
    output_dir='av_agent_output'
    #[5172,6564,7043,7483,8268]#
    llm_name='codellama:13b'
    view_train_1='view_train_1'
    view_train_2='view_train_2'
    view_train_3='view_train_3'
    view_train_4='view_train_4'
    seqs_train = test001.load_seqs_from_file(train_file)
    view_test_1='view_test_1'
    view_test_2='view_test_2'
    view_test_3='view_test_3'
    view_test_4='view_test_4'
    view_test_5='view_test_5'
    seqs_test = test001.load_seqs_from_file(test_file)
    ml_pre_dict,acc_ml,pre_ml,rec_ml,f1_ml=test001.main_abla(train_file,seqs_test)
    # test002.main(seqs_train,out_file='llm_features_train.json')#可疑代码提取
    # test003.main(seqs_train,output_dir,llm_name,out_file='llm_features_train.json')#模板生成
    # #生成测试集模板
    # test002.main(seqs_test,out_file='llm_features_test.json')#可疑代码提取
    # test003.main(seqs_test,output_dir,llm_name,out_file='llm_features_test.json')#模板生成

    with ThreadPoolExecutor(max_workers=3) as executor:
        # 所有任务列表
        tasks = [
            (test004_1.main,seqs_train,output_dir,view_train_1,llm_name),
            (test004_2.main,seqs_train,output_dir,view_train_2,llm_name),
            (test004_3.main,seqs_train,output_dir,view_train_3,llm_name),
            (test004_4.main,seqs_train,output_dir,view_train_4,llm_name),
            (test004_1.main,seqs_test,output_dir,view_test_1,llm_name)  ,
            (test004_2.main,seqs_test,output_dir,view_test_2,llm_name)  ,
            (test004_3.main,seqs_test,output_dir,view_test_3,llm_name)  ,
            (test004_4.main,seqs_test,output_dir,view_test_4,llm_name)  ,
            (test004_4_clear_ml.main,seqs_test,output_dir,view_test_5,llm_name)
        ]
        futures = [executor.submit(func, *args) for func, *args in tasks]
        for f in futures:
            f.result()



    #训练机器学习模型
    model_path = r"/home/changxiaosong/python/malwareTest/deepseek-coder-1.3b-base"
    train_data={}
    conn = get_connection()
    for one in seqs_train:
        file_path=f'seq_{one}_interpretability.json'
        if os.path.exists(view_train_1+os.sep+file_path) and \
                os.path.exists(view_train_2+os.sep+file_path) and \
                os.path.exists(view_train_3+os.sep+file_path) :
            train_data[one] = {}
            train_data[one]['label']=get_label_by_seq(conn,one)
            train_data[one][view_train_1]=get_json_content(view_train_1,one)
            train_data[one][view_train_2]=get_json_content(view_train_2,one)
            train_data[one][view_train_3]=get_json_content(view_train_3,one)
    test_data={}
    conn = get_connection()
    ml_seqs={}
    for one in seqs_test:
        file_path=f'seq_{one}_interpretability.json'
        if os.path.exists(view_test_1+os.sep+file_path) and\
            os.path.exists(view_test_2+os.sep+file_path) and\
            os.path.exists(view_test_3+os.sep+file_path) :
            test_data[one] = {}
            test_data[one]['label']=get_label_by_seq(conn,one)
            test_data[one][view_test_1]=get_json_content(view_test_1,one)
            test_data[one][view_test_2]=get_json_content(view_test_2,one)
            test_data[one][view_test_3]=get_json_content(view_test_3,one)
        else:
            print(f'{one}没有模板，维持原ML结果')
            ml_seqs[one]={'pre':ml_pre_dict[one],'true_label':get_label_by_seq(conn,one)}
    conn.close()



    X_train, y_train, X_test, y_test = prepare_data(train_data, test_data)
    # 使用微调方式训练
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("开始微调训练...")
    finetuned_model = train_model_finetune(X_train, y_train, model_path, device)


    #检测测试集模板
    print("\n模型评估...")
    y_pred = finetuned_model.predict(X_test)
    #对于无LLM结果的一部分seq，补充
    for key in ml_seqs:
        y_pred.append(ml_seqs[key]['pre'])
        y_test.append(ml_seqs[key]['true_label'])
    #输出检测结果

    print(['llm-result ML',acc_ml,pre_ml,rec_ml,f1_ml])
    acc,pre,rec,f1=get_pre_result_one(view_test_5)
    print(['llm-result LLM',acc,pre,rec,f1])
    acc,pre,rec,f1=get_pre_result_one(view_test_4)
    print(['llm-result ML+LLM',acc,pre,rec,f1])
    acc,pre,rec,f1=get_pre_result(view_test_1,view_test_2,view_test_3)
    print(['llm-result ML+LLM+MV',acc,pre,rec,f1])
    balance_accuracy_llm, precision_llm, recall_llm, f1_value_llm = \
        plot_value(y_test, y_pred)
    print(['llm-result ML+LLM+MV+Correct',balance_accuracy_llm, precision_llm, recall_llm, f1_value_llm])
    get_final_csv(ml_pre_dict,view_test_1,view_test_2,view_test_3)
    import csv
    with open('pre_label.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seq', 'true_label', 'llm_label', 'ml_label'])
        for seq, true_label, llm_label in zip(seqs_test, y_test, y_pred):
            writer.writerow([seq, true_label, llm_label, ml_pre_dict.get(seq, '')])


    #去除多视角
    X_train, y_train, X_test, y_test = prepare_data_not_mv(train_data, test_data)
    finetuned_model = train_model_finetune(X_train, y_train, model_path, device)


    #检测测试集模板
    print("\n模型评估...")
    y_pred = finetuned_model.predict(X_test)
    #对于无LLM结果的一部分seq，补充
    for key in ml_seqs:
        y_pred.append(ml_seqs[key]['pre'])
        y_test.append(ml_seqs[key]['true_label'])
    #输出检测结果
    acc, pre, rec, f1 = \
        plot_value(y_test, y_pred)
    print(['llm-result ML+LLM+Correct',acc, pre, rec, f1])

