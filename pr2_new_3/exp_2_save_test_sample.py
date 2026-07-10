# save_test_labels.py
import numpy as np
import os

import torch

from exp_1 import load_methods_from_json, Config, DeepSeekEncoder, ContrastiveModel
from sklearn.model_selection import train_test_split

def save_test_labels():
    # 加载数据
    methods, labels, _ = load_methods_from_json(Config.JSON_FILE)

    # 划分数据集
    _, test_methods, _, test_labels, _, _ = train_test_split(
        methods, labels, range(len(labels)),
        test_size=0.1, random_state=Config.RANDOM_SEED, stratify=labels
    )

    # 编码测试集
    encoder = DeepSeekEncoder(Config.MODEL_PATH)
    test_features = encoder.encode(test_methods)

    # 加载对比学习模型并获取嵌入
    model = ContrastiveModel(
        input_dim=test_features.shape[1],
        hidden_dim=Config.HIDDEN_DIM,
        output_dim=Config.OUTPUT_DIM
    )

    if os.path.exists("visualizations/contrastive_model.pth"):
        model.load_state_dict(torch.load("visualizations/contrastive_model.pth"))
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        with torch.no_grad():
            test_tensor = torch.FloatTensor(test_features).to(device)
            test_embeddings = model(test_tensor).cpu().numpy()
    else:
        test_embeddings = test_features

    # 保存
    np.save("visualizations/test_contrastive_embeddings.npy", test_embeddings)
    np.save("visualizations/test_labels.npy", np.array(test_labels))

    print(f"测试集嵌入保存: {test_embeddings.shape}")
    print(f"测试集标签保存: {len(test_labels)}个")

if __name__ == "__main__":
    save_test_labels()