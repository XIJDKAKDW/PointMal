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
import os
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import torch
import warnings

warnings.filterwarnings('ignore')
import platform
import sys
system = platform.system()

if system == "Linux":
    sys.path.append(r"/home/changxiaosong/python/malwareTest")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr2_final")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr2_new_2")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr3")
from pr2_new_3 import test001, test002, test003, test004_1, test004_2, test004_3
from pr2_new_3.test001 import get_label_by_seq, get_connection, plot_value

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

def get_vec(X_train,  X_test):
    """训练模型"""
    # 加载DeepSeek模型
    print("加载DeepSeek模型...")
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True, help='训练集文件路径')
    parser.add_argument('--test', required=True, help='测试集文件路径')
    args = parser.parse_args()
    # llm_names=['gpt-oss:20b','gemma3:1b','gemma3:4b','codellama:13b','codellama:7b','deepseek-coder:1.3b','deepseek-coder-v2:16b']
    llm_names=['gemma3:1b','gemma3:4b','codellama:13b','codellama:7b','deepseek-coder:1.3b','deepseek-coder-v2:16b']
    train_file = args.train
    test_file = args.test
    output_dir='av_agent_output'
    view_train_1='view_train_1'
    view_train_2='view_train_2'
    view_train_3='view_train_3'
    seqs_train = test001.load_seqs_from_file(train_file)
    view_test_1='view_test_1'
    view_test_2='view_test_2'
    view_test_3='view_test_3'
    seqs_test = test001.load_seqs_from_file(test_file)
    ml_pre_dict=test001.main(train_file,seqs_test)
    for llm_name in llm_names:
        # if llm_name=='gpt-oss:20b':
        #     test002.main(seqs_train,out_file='llm_features_train.json')#可疑代码提取
        #     test003.main(seqs_train,output_dir,llm_name,out_file='llm_features_train.json')#模板生成
        #     # # 生成测试集模板
        #     test002.main(seqs_test,out_file='llm_features_test.json')#可疑代码提取
        #     test003.main(seqs_test,output_dir,llm_name,out_file='llm_features_test.json')#模板生成
        with ThreadPoolExecutor(max_workers=3) as executor:
            tasks = [
                (test004_1.main,seqs_train,output_dir,view_train_1,llm_name),
                (test004_2.main,seqs_train,output_dir,view_train_2,llm_name),
                (test004_3.main,seqs_train,output_dir,view_train_3,llm_name),
                # 生成测试集模板
                (test004_1.main,seqs_test,output_dir,view_test_1,llm_name),
                (test004_2.main,seqs_test,output_dir,view_test_2,llm_name),
                (test004_3.main,seqs_test,output_dir,view_test_3,llm_name)
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
        X_train,  X_test=get_vec(X_train,  X_test)
        mlp=train_model(X_train, y_train)

        #检测测试集模板
        print("\n模型评估...")
        y_pred = mlp.predict(X_test)
        #对于无LLM结果的一部分seq，补充
        for key in ml_seqs:
            y_pred.append(ml_seqs[key]['pre'])
            y_test.append(ml_seqs[key]['true_label'])
        #输出检测结果
        balance_accuracy_llm, precision_llm, recall_llm, f1_value_llm = \
            plot_value(y_test, y_pred)
        print(['llm-result：robustness',llm_name,balance_accuracy_llm, precision_llm, recall_llm, f1_value_llm])

