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

def extract_features(data_dict, view_keys):
    """从数据字典中提取特征和标签"""
    X = []
    y = []

    for seq, data in data_dict.items():
        combined_text = f""
        # combined_text += f"=== CHANNEL_1 ===\n{data.get(view_keys[0], '')}\n\n"
        # combined_text += f"=== CHANNEL_2 ===\n{data.get(view_keys[1], '')}\n\n"
        combined_text += f"=== CHANNEL_3 ===\n{data.get(view_keys[2], '')}"
        X.append(combined_text)
        y.append(data['label'])
    return X, y
def prepare_data(train_data, test_data):
    print("准备训练数据...")
    X_train, y_train = extract_features(train_data, [view_train_1, view_train_2, view_train_3])
    print("准备测试数据...")
    X_test, y_test = extract_features(test_data, [view_test_1, view_test_2, view_test_3])
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

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class JointFineTuneModel(nn.Module):
    def __init__(self, llm_path, freeze_llm=True):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(llm_path, trust_remote_code=True)
        self.hidden_size = self.llm.config.hidden_size

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.llm(input_ids, attention_mask, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        # Mean pooling
        mask = attention_mask.unsqueeze(-1)
        embedding = (hidden * mask).sum(1) / mask.sum(1)
        return self.classifier(embedding)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=2048):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True,
                                  padding='max_length', max_length=self.max_len,
                                  return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def train_model_finetune(X_train, y_train, model_path,cuda=0):
    """使用微调方式训练模型，返回包装后的模型对象"""
    # 加载tokenizer
    device = torch.device('cuda:'+str(cuda) if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 创建训练数据集和数据加载器
    train_dataset = TextDataset(X_train, y_train, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # 阶段1: 冻结LLM，只训练分类头
    print("="*50)
    print("阶段1: 冻结LLM，训练分类头")
    model = JointFineTuneModel(model_path, freeze_llm=True).to(device)

    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}")

    # 阶段2: 解冻最后2层LLM，联合微调
    print("\n" + "="*50)
    print("阶段2: 解冻最后2层，联合微调")
    for param in model.llm.parameters():
        param.requires_grad = False

    # 解冻最后2层
    for name, param in model.llm.named_parameters():
        if 'layers' in name:
            layer_num = int(name.split('.')[2])
            if layer_num >= model.llm.config.num_hidden_layers - 2:
                param.requires_grad = True

    optimizer = optim.AdamW([
        {'params': model.classifier.parameters(), 'lr': 1e-4},
        {'params': model.llm.parameters(), 'lr': 1e-5}
    ])

    for epoch in range(5):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}")

    # 返回包装后的模型
    return FineTunedModelWrapper(model, tokenizer, device)
class FineTunedModelWrapper:
    """包装微调后的模型，提供predict接口"""
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def predict(self, texts):
        """预测文本列表的标签"""
        self.model.eval()
        predictions = []

        # 创建Dataset和DataLoader
        dataset = TextDataset(texts, [0]*len(texts), self.tokenizer)  # 标签占位
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                logits = self.model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)

        return np.array(predictions)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True, help='训练集文件路径')
    parser.add_argument('--test', required=True, help='测试集文件路径')
    parser.add_argument('--output_file', required=True, help='输出标签文件')
    parser.add_argument('--cuda', required=True, help='cuda')


    args = parser.parse_args()

    train_file = args.train
    test_file = args.test
    output_file=args.output_file
    cuda=args.cuda
    output_dir='av_agent_output'
    #[5172,6564,7043,7483,8268]#
    llm_name='codellama:13b'
    view_train_1='view_train_1'
    view_train_2='view_train_2'
    view_train_3='view_train_3'
    seqs_train = test001.load_seqs_from_file(train_file)
    view_test_1='view_test_1'
    view_test_2='view_test_2'
    view_test_3='view_test_3'


    seqs_test = test001.load_seqs_from_file(test_file)
    ml_pre_dict=test001.main(train_file,seqs_test)
    # test002.main(seqs_train,out_file='llm_features_train.json')#可疑代码提取
    # test003.main(seqs_train,output_dir,llm_name,out_file='llm_features_train.json')#模板生成
    # # 生成测试集模板
    # test002.main(seqs_test,out_file='llm_features_test.json')#可疑代码提取
    # test003.main(seqs_test,output_dir,llm_name,out_file='llm_features_test.json')#模板生成


    # test004_1.main(seqs_train,output_dir,view_train_1,llm_name)
    # test004_2.main(seqs_train,output_dir,view_train_2,llm_name)
    # test004_3.main(seqs_train,output_dir,view_train_3,llm_name)
    # test004_1.main(seqs_test,output_dir,view_test_1,llm_name)
    # test004_2.main(seqs_test,output_dir,view_test_2,llm_name)
    # test004_3.main(seqs_test,output_dir,view_test_3,llm_name)
    # 训练机器学习模型
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
        if os.path.exists(view_test_1+os.sep+file_path) and \
                os.path.exists(view_test_2+os.sep+file_path) and \
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
    print("30秒后开始微调训练...")
    import time
    time.sleep(30)
    finetuned_model = train_model_finetune(X_train, y_train, model_path,cuda)


    #检测测试集模板
    print("\n模型评估...")
    y_pred = finetuned_model.predict(X_test)
    y_pred = list(y_pred)
    y_test = list(y_test)  # 如果 y_test 是数组

    #对于无LLM结果的一部分seq，补充
    # for key in ml_seqs:
    #     y_pred.append(ml_seqs[key]['pre'])
    #     y_test.append(ml_seqs[key]['true_label'])
    #
    print(y_test)
    print(y_pred)
    #输出检测结果
    balance_accuracy_llm, precision_llm, recall_llm, f1_value_llm = \
        plot_value(y_test, y_pred)
    print(['llm-result：',balance_accuracy_llm, precision_llm, recall_llm, f1_value_llm])
    get_final_csv(ml_pre_dict,view_test_1,view_test_2,view_test_3)
    import csv
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seq', 'true_label', 'llm_label', 'ml_label'])
        for seq, true_label, llm_label in zip(seqs_test, y_test, y_pred):
            writer.writerow([seq, true_label, llm_label, ml_pre_dict.get(seq, '')])

