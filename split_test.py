import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.datasets import Coauthor
from torch_geometric.data import Data

# 修改 split_test.py
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data

def load_recommendation_dataset(name="sparse_yelp"):
    """
    加载稀疏矩阵格式的推荐数据集（yelp, amazon, gowalla等）
    数据集包含三个文件：trnMat.pkl, valMat.pkl, tstMat.pkl
    每个文件是一个scipy.sparse矩阵，形状为(num_users, num_items)
    """
    import pickle
    import scipy.sparse as sp
    
    dataset_path = f"dataset/{name}"
    
    # 加载训练/验证/测试矩阵
    with open(f"{dataset_path}/trnMat.pkl", 'rb') as f:
        train_mat = pickle.load(f)
    with open(f"{dataset_path}/valMat.pkl", 'rb') as f:
        val_mat = pickle.load(f)
    with open(f"{dataset_path}/tstMat.pkl", 'rb') as f:
        test_mat = pickle.load(f)
    
    num_users, num_items = train_mat.shape
    print(f"训练数据集: {name}, 用户数: {num_users}, 物品数: {num_items}")
    
    # 【修复数据泄露】只使用训练集交互构建图的边索引
    # 验证集和测试集的交互不应该出现在图结构中，否则模型会通过消息传递学习到这些信息
    train_mat_coo = train_mat.tocoo()
    
    # 构建边索引（用户->物品，物品索引需要加上num_users）
    # 注意：这里只使用训练集的边！
    edges = []
    for i in range(len(train_mat_coo.row)):
        u = train_mat_coo.row[i]  # 用户ID (0 ~ num_users-1)
        v = train_mat_coo.col[i] + num_users  # 物品ID (num_users ~ num_users+num_items-1)
        edges.append([u, v])
        edges.append([v, u])  # 添加反向边，构建无向图
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    print(f"图构建信息：仅使用训练集边，边数量: {edge_index.shape[1]} (= 2 × {len(train_mat_coo.row)} 训练交互)")
    
    # 生成节点特征（随机初始化，实际应用中可用用户/物品属性）
    # user_features = torch.randn(num_users, 64)
    # item_features = torch.randn(num_items, 64)
    # x = torch.cat([user_features, item_features], dim=0)
    
    # 标记节点类型（0=用户，1=物品）
    node_types = torch.cat([
        torch.zeros(num_users, dtype=torch.long),
        torch.ones(num_items, dtype=torch.long)
    ])
    
    # 提取训练/验证/测试集的交互对
    def extract_interactions(mat):
        mat_coo = mat.tocoo()
        interactions = []
        for i in range(len(mat_coo.row)):
            u = mat_coo.row[i]
            v = mat_coo.col[i]
            interactions.append([u, v])
        return np.array(interactions)
    
    train_interactions = extract_interactions(train_mat)
    val_interactions = extract_interactions(val_mat)
    test_interactions = extract_interactions(test_mat)
    
    # 合并所有交互并创建mask
    all_interactions = np.vstack([train_interactions, val_interactions, test_interactions])
    num_train = len(train_interactions)
    num_val = len(val_interactions)
    num_test = len(test_interactions)
    total = num_train + num_val + num_test
    
    train_mask = torch.zeros(total, dtype=torch.bool)
    val_mask = torch.zeros(total, dtype=torch.bool)
    test_mask = torch.zeros(total, dtype=torch.bool)
    train_mask[:num_train] = True
    val_mask[num_train:num_train+num_val] = True
    test_mask[num_train+num_val:] = True
    
    print(f"训练交互数: {num_train}, 验证交互数: {num_val}, 测试交互数: {num_test}")
    
    data = Data(
        # x=x,
        edge_index=edge_index,
        node_types=node_types,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        interactions=torch.tensor(all_interactions, dtype=torch.long)
    )
    return data

# def load_recommendation_dataset(name="ml-100k", split="normal"):
#     """
#     加载推荐数据集（用户-物品二部图）
#     格式：用户ID、物品ID、交互标签（1表示有交互，0表示无）
#     """
#     # 如果是稀疏数据集（yelp, amazon, gowalla），使用专门的加载函数
#     if name.startswith("sparse_"):
#         return load_sparse_dataset(name)
    
#     # 示例：加载MovieLens-100k数据
#     if name == "ml-100k":
#         data_path = "F:/code/NLPshiyan/ml-100k/ml-100k/u.data"
#         columns = ['user_id', 'item_id', 'rating', 'timestamp']
#         df = pd.read_csv(data_path, sep='\t', names=columns)
        
#         # 提取用户-物品交互（二值化，1表示有交互）
#         df['interaction'] = 1
#         user_ids = df['user_id'].unique()
#         item_ids = df['item_id'].unique()
        
#         # 节点ID映射（用户ID从0开始，物品ID接在用户后）
#         user_id_map = {u: i for i, u in enumerate(user_ids)}
#         item_id_map = {i: len(user_ids) + j for j, i in enumerate(item_ids)}
#         num_users = len(user_ids)
#         num_items = len(item_ids)
        
#         # 构建边索引（用户->物品）
#         edges = []
#         for _, row in df.iterrows():
#             u = user_id_map[row['user_id']]
#             v = item_id_map[row['item_id']]
#             edges.append((u, v))
#         edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
#         # 生成节点特征（示例：随机特征，实际可用用户/物品属性）
#         user_features = torch.randn(num_users, 64)  # 假设用户特征维度64
#         item_features = torch.randn(num_items, 64)  # 假设物品特征维度64
#         x = torch.cat([user_features, item_features], dim=0)
        
#         # 标记节点类型（0=用户，1=物品）
#         node_types = torch.cat([
#             torch.zeros(num_users, dtype=torch.long),
#             torch.ones(num_items, dtype=torch.long)
#         ])
        
#         # 划分训练/验证/测试集（按交互划分）
#         # interactions = df[['user_id', 'item_id']].values
#         mapped_interactions = []
#         for _, row in df.iterrows():
#             u = user_id_map[row['user_id']]  # 映射后的用户索引（0~num_users-1）
#             v = item_id_map[row['item_id']] - num_users  # 映射后的物品相对索引（0~num_items-1）
#             mapped_interactions.append((u, v))
#         interactions = np.array(mapped_interactions)
#         np.random.shuffle(interactions)
#         train_size = int(0.8 * len(interactions))
#         val_size = int(0.1 * len(interactions))
        
#         train_mask = torch.zeros(len(interactions), dtype=torch.bool)
#         val_mask = torch.zeros(len(interactions), dtype=torch.bool)
#         test_mask = torch.zeros(len(interactions), dtype=torch.bool)
#         train_mask[:train_size] = True
#         val_mask[train_size:train_size+val_size] = True
#         test_mask[train_size+val_size:] = True
        
#         data = Data(
#             x=x,
#             edge_index=edge_index,
#             node_types=node_types,  # 新增：标记用户/物品类型
#             train_mask=train_mask,
#             val_mask=val_mask,
#             test_mask=test_mask,
#             interactions=torch.tensor(interactions, dtype=torch.long)  # 存储原始交互对
#         )
#         return data
#     else:
#         raise ValueError(f"不支持的推荐数据集: {name}")

