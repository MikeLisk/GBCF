import os
import argparse
from gb_division import gb_division
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
from tools.split_indices import split_indices
from split_test import load_recommendation_dataset
# from models.APPNP import appnp_c
from models.RecGCN import RecGCN
import copy
from sklearn.metrics import ndcg_score
import torch.nn as nn
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# class GCN(torch.nn.Module):
#     def __init__(self, args):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(args.num_features, args.hidden)
#         self.conv2 = GCNConv(args.hidden, args.num_classes)

#     def reset_parameters(self):
#         self.conv1.reset_parameters()
#         self.conv2.reset_parameters()

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x,training= self.training)
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=1)

def bpr_loss(positive_scores, negative_scores):
    """BPR损失：最大化正样本分数，最小化负样本分数"""
    # 添加数值稳定性：限制分数差值范围，避免sigmoid溢出
    diff = torch.clamp(positive_scores - negative_scores, min=-10, max=10)
    loss = -torch.mean(F.logsigmoid(diff))  # 使用logsigmoid更稳定
    return loss



def evaluate_full_ranking(user_emb, item_emb, test_interactions, train_interactions, 
                         num_items, K=20, device='cpu', debug=False):
    """
    全物品排序评估（标准推荐系统评估方法）
    
    对每个测试用户，在过滤掉训练集物品后，对所有剩余物品进行排序，
    计算测试集正样本在排序中的位置，得到准确的 Hit@K, MRR, NDCG@K。
    
    Args:
        user_emb: 用户嵌入 [num_users, embedding_dim]
        item_emb: 物品嵌入 [num_items, embedding_dim]
        test_interactions: 测试集交互 [num_test_interactions, 2]
        train_interactions: 训练集交互 [num_train_interactions, 2]
        num_items: 物品总数
        K: 评估的K值（如 K=10 计算 Hit@10, NDCG@10）
        device: 设备
        debug: 是否输出调试信息
    
    Returns:
        hit_ratio: Hit@K (测试集正样本在top-K中的比例)
        mrr: 平均倒数排名
        ndcg: NDCG@K (标准推荐系统NDCG)
    """
    # 构建训练集交互字典（用于过滤）
    user_train_items = {}
    for interaction in train_interactions:
        user_id = interaction[0].item() if torch.is_tensor(interaction[0]) else interaction[0]
        item_id = interaction[1].item() if torch.is_tensor(interaction[1]) else interaction[1]
        if user_id not in user_train_items:
            user_train_items[user_id] = set()
        user_train_items[user_id].add(item_id)
    
    # 构建测试集用户的正样本列表
    test_user_items = {}
    for interaction in test_interactions:
        user_id = interaction[0].item() if torch.is_tensor(interaction[0]) else interaction[0]
        item_id = interaction[1].item() if torch.is_tensor(interaction[1]) else interaction[1]
        if user_id not in test_user_items:
            test_user_items[user_id] = []
        test_user_items[user_id].append(item_id)
    
    hit_count = 0
    reciprocal_ranks = []
    ndcg_scores = []
    total_users = 0
    
    # 【方案四：验证数据泄漏修复】调试信息统计
    debug_info = {
        'total_items': num_items,
        'avg_train_items_filtered': [],
        'avg_candidate_items': [],
        'test_items_in_train': 0,  # 测试物品出现在训练集的次数（应该为0）
    }
    
    # 对每个测试用户进行评估
    for user_id, pos_items in test_user_items.items():
        if len(pos_items) == 0:
            continue
        
        total_users += 1
        
        # 计算用户对所有物品的分数
        user_embedding = user_emb[user_id].unsqueeze(0)  # [1, dim]
        all_scores = torch.sum(user_embedding * item_emb, dim=1)  # [num_items]
        
        # 过滤掉训练集中的物品（将分数设为极小值）
        train_items = user_train_items.get(user_id, set())
        
        # 【验证】检查测试集物品是否在训练集中（应该不在）
        for test_item in pos_items:
            if test_item in train_items:
                debug_info['test_items_in_train'] += 1
        
        # 记录过滤的训练集物品数量
        debug_info['avg_train_items_filtered'].append(len(train_items))
        debug_info['avg_candidate_items'].append(num_items - len(train_items))
        
        for train_item in train_items:
            all_scores[train_item] = float('-inf')
        
        # 排序（降序）
        sorted_indices = torch.argsort(all_scores, descending=True)
        top_k_items = sorted_indices[:K].cpu().numpy()
        
        # 计算 Hit@K（至少有一个正样本在top-K中）
        if any(item in pos_items for item in top_k_items):
            hit_count += 1
        
        # 计算 MRR（第一个正样本的排名倒数）
        for rank, item_id in enumerate(sorted_indices.cpu().numpy(), 1):
            if item_id in pos_items:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
        
        # 计算 NDCG@K（标准公式）
        dcg = 0.0
        for rank, item_id in enumerate(top_k_items, 1):
            if item_id in pos_items:
                dcg += 1.0 / np.log2(rank + 1)
        
        # IDCG（理想情况：所有正样本都在前面）
        idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(pos_items), K))])
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)
    
    # 【方案四：输出调试信息】
    if debug:
        avg_train_filtered = np.mean(debug_info['avg_train_items_filtered']) if debug_info['avg_train_items_filtered'] else 0
        avg_candidates = np.mean(debug_info['avg_candidate_items']) if debug_info['avg_candidate_items'] else 0
        print(f"\n【评估函数调试信息 - 验证数据泄漏修复】")
        print(f"  总物品数: {debug_info['total_items']}")
        print(f"  平均每用户过滤的训练集物品数: {avg_train_filtered:.1f}")
        print(f"  平均每用户候选物品数: {avg_candidates:.1f} ({100*avg_candidates/num_items:.1f}%)")
        print(f"  测试物品出现在训练集的次数: {debug_info['test_items_in_train']} (应为0)")
        if debug_info['test_items_in_train'] > 0:
            print(f"  ⚠️ 警告：检测到数据泄漏！测试集物品出现在了训练集中！")
        else:
            print(f"  ✓ 数据泄漏检查通过：测试集和训练集物品无重叠")
        print()
    
    # 计算平均指标
    hit_ratio = hit_count / total_users if total_users > 0 else 0.0
    mrr = np.mean(reciprocal_ranks) if len(reciprocal_ranks) > 0 else 0.0
    avg_ndcg = np.mean(ndcg_scores) if len(ndcg_scores) > 0 else 0.0
    
    return hit_ratio, mrr, avg_ndcg


def main():
    #parameter settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="sparse_yelp")
    # parser.add_argument('--split_data', type=str, default="normal")
    # parser.add_argument('--runs', type=int, default=20)
    # parser.add_argument('--models', type=str, default='GCN')
    # parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    # parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)  # 降低学习率以提高稳定性
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    # parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--ball_r', type=float, default=0.5)
    # parser.add_argument('--noise', type=int, default=0)
    parser.add_argument('--K', type=int, default=20)
    # parser.add_argument('--alpha', type=float, default=0.1)
    # parser.add_argument('--heads', type=int, default=8)
# ... 参数设置 ...
    parser.add_argument('--embedding_dim', type=int, default=128)  # 新增：嵌入维度
    # parser.add_argument('--num_users', type=int, default=943)  # 将自动从数据集中获取
    # parser.add_argument('--k', type=int, default=20)  # 评估指标@K
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 打印参数设置信息
    print("\n" + "=" * 80)
    print("Parameter Settings")
    print("=" * 80)
    for arg in vars(args):
        print(f"{arg:<20} : {getattr(args, arg)}")
    print(f"{'device':<20} : {device}")
    print("=" * 80 + "\n", flush=True)
    # 加载推荐数据集
    print("Loading data...", flush=True)
    data = load_recommendation_dataset(name=args.dataset)
    # args.num_features = data.x.shape[1]
    args.num_users = (data.node_types == 0).sum().item()  # 从数据中获取用户数

    # 粒球划分（复用原有逻辑）
    new_data = gb_division(data, args)
    # args.num_features = new_data['gb_features'].shape[1]  
    args.num_features = args.embedding_dim # 既然没有原始特征，GCN输入维度设为embedding_dim（或任意值，因为RecGCN有embedding层）
    
    # 初始化推荐模型
    num_gb = new_data['gb_features'].shape[0]  # 粒球数量
    print(f"启用可学习粒球嵌入，粒球数量: {num_gb}, 特征维度: {args.num_features}")
    model = RecGCN(args, num_gb_nodes=num_gb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 训练循环
    print(f"开始训练，共 {args.epochs} 轮...")
    print("=" * 80)
    
    # 【数据验证】确保没有数据泄露
    print("\n数据验证信息：")
    print(f"  原始数据边数量: {data.edge_index.shape[1]}")
    print(f"  粒球图边数量: {len(new_data['adj'][0])}")
    print(f"  训练交互数: {data.train_mask.sum().item()}")
    print(f"  验证交互数: {data.val_mask.sum().item()}")
    print(f"  测试交互数: {data.test_mask.sum().item()}")
    expected_edges = 2 * data.train_mask.sum().item()
    actual_edges = data.edge_index.shape[1]
    if actual_edges == expected_edges:
        print(f"  ✓ 边数量验证通过: {actual_edges} == 2 × {data.train_mask.sum().item()}")
    else:
        print(f"  ✗ 警告：边数量不匹配！期望 {expected_edges}，实际 {actual_edges}")
        print(f"    可能存在数据泄露！")
    print("=" * 80)
    print()
    
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        # 从字典中提取粒球特征和边索引，并转换为torch张量
        x = torch.tensor(new_data['gb_features'], dtype=torch.float32).to(device)  # 关键修改
        edge_index = torch.tensor(new_data['adj'], dtype=torch.long).to(device)  # 关键修改
        embeddings = model(x, edge_index)  # 所有粒球节点的嵌入
        # L2归一化嵌入向量，防止数值过大
        embeddings = F.normalize(embeddings, p=2, dim=1)
        # user_emb = embeddings[:args.num_users]  # 分离用户嵌入
        # item_emb = embeddings[args.num_users:]  # 分离物品嵌入
        # 获取原始节点到粒球的映射
        node_to_gb = new_data['node_to_gb']  # 原始节点ID -> 粒球索引
        num_original_nodes = len(data.node_types)  # 原始节点总数（用户+物品）
        num_users = args.num_users
        num_items = num_original_nodes - num_users  # 原始物品数量

        # 映射原始用户嵌入（聚合其所属粒球的嵌入）
        # 映射原始用户嵌入（聚合其所属所有粒球的嵌入）
        user_emb = torch.zeros((num_users, args.embedding_dim), device=device)
        for user_id in range(num_users):
            gb_indices = node_to_gb.get(user_id, [])  # 获取用户所属的所有粒球索引
            if gb_indices:
                # 多粒球嵌入均值聚合
                user_emb[user_id] = torch.mean(embeddings[gb_indices], dim=0)

        # 映射原始物品嵌入
        item_emb = torch.zeros((num_items, args.embedding_dim), device=device)
        for item_id in range(num_items):
            original_node_id = num_users + item_id
            gb_indices = node_to_gb.get(original_node_id, [])
            if gb_indices:
                item_emb[item_id] = torch.mean(embeddings[gb_indices], dim=0)
        
        # 采样正负样本（训练集中的交互为正，随机未交互为负）
        train_interactions = data.interactions[data.train_mask]
        pos_user_ids = train_interactions[:, 0]
        pos_item_ids = train_interactions[:, 1]
        # 生成负样本前添加检查
        if item_emb.shape[0] == 0:
            raise ValueError("物品嵌入数量为0，请检查原始节点到粒球的映射是否正确")
        user_pos_items = {}
        for u, i in train_interactions:
            u = u.item()
            i = i.item()
            if u not in user_pos_items:
                user_pos_items[u] = set()
            user_pos_items[u].add(i)
        
        # 生成负样本（确保不在正样本中）
        neg_item_ids = []
        for u in pos_user_ids:
            u = u.item()
            while True:
                neg_i = torch.randint(0, item_emb.shape[0], (1,), device=device).item()
                if neg_i not in user_pos_items.get(u, set()):
                    neg_item_ids.append(neg_i)
                    break
        neg_item_ids = torch.tensor(neg_item_ids, device=device)
        # neg_item_ids = torch.randint(0, item_emb.shape[0], (len(pos_user_ids),), device=device)  # 随机负样本

        # 计算正负样本分数
        pos_scores = model.predict(user_emb, item_emb, pos_user_ids, pos_item_ids)
        neg_scores = model.predict(user_emb, item_emb, pos_user_ids, neg_item_ids)

        # BPR损失
        loss = bpr_loss(pos_scores, neg_scores)
        
        # 检查loss是否有效
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"警告: Epoch {epoch+1} 出现无效loss，跳过此batch")
            continue
        
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 验证
        if (epoch + 1) % 5 == 0 or epoch == 0:  # 每5轮或第一轮进行验证
            model.eval()
            with torch.no_grad():
                val_x = torch.tensor(new_data['gb_features'], dtype=torch.float32).to(device)
                val_edge_index = torch.tensor(new_data['adj'], dtype=torch.long).to(device)
                val_embeddings = model(val_x, val_edge_index)  # 用粒球数据计算嵌入
                # L2归一化
                val_embeddings = F.normalize(val_embeddings, p=2, dim=1)
                
                # 修复：使用node_to_gb映射获取原始用户和物品嵌入（与训练逻辑一致）
                node_to_gb_val = new_data['node_to_gb']
                num_original_nodes_val = len(data.node_types)
                num_users_val = args.num_users
                num_items_val = num_original_nodes_val - num_users_val
                
                # 映射原始用户嵌入（多粒球均值聚合，与训练逻辑一致）
                val_user_emb = torch.zeros((num_users_val, args.embedding_dim), device=device)
                user_mapped_count = 0
                for user_id in range(num_users_val):
                    gb_indices = node_to_gb_val.get(user_id, [])  # 获取用户所属的所有粒球索引（列表）
                    if gb_indices:
                        # 多粒球嵌入均值聚合
                        val_user_emb[user_id] = torch.mean(val_embeddings[gb_indices], dim=0)
                        user_mapped_count += 1
                
                # 映射原始物品嵌入（多粒球均值聚合，与训练逻辑一致）
                val_item_emb = torch.zeros((num_items_val, args.embedding_dim), device=device)
                item_mapped_count = 0
                for item_id in range(num_items_val):
                    original_node_id = num_users_val + item_id
                    gb_indices = node_to_gb_val.get(original_node_id, [])  # 获取物品所属的所有粒球索引（列表）
                    if gb_indices:
                        val_item_emb[item_id] = torch.mean(val_embeddings[gb_indices], dim=0)
                        item_mapped_count += 1
                
                # 获取验证集交互和训练集交互（需要在调试代码之前定义）
                val_interactions = data.interactions[data.val_mask]
                train_interactions = data.interactions[data.train_mask]
                
                # 【调试】首次验证时输出映射统计信息
                if epoch == 0:
                    print(f"\n【调试信息 - 验证阶段】")
                    print(f"  用户映射覆盖率: {user_mapped_count}/{num_users_val} ({100*user_mapped_count/num_users_val:.1f}%)")
                    print(f"  物品映射覆盖率: {item_mapped_count}/{num_items_val} ({100*item_mapped_count/num_items_val:.1f}%)")
                    print(f"  用户嵌入非零比例: {(val_user_emb.abs().sum(dim=1) > 0).sum().item()}/{num_users_val}")
                    print(f"  物品嵌入非零比例: {(val_item_emb.abs().sum(dim=1) > 0).sum().item()}/{num_items_val}")
                    print(f"  粒球总数: {len(val_embeddings)}")
                    print(f"  node_to_gb键数量: {len(node_to_gb_val)}")
                    # 检查验证集用户是否有映射
                    val_user_ids = val_interactions[:, 0].unique().cpu().numpy()
                    val_users_mapped = sum(1 for uid in val_user_ids if node_to_gb_val.get(uid, []))
                    print(f"  验证集用户映射: {val_users_mapped}/{len(val_user_ids)} ({100*val_users_mapped/len(val_user_ids):.1f}%)")
                

                if len(val_interactions) > 0:
                    # 使用全物品排序评估（标准推荐系统评估方法）
                    # 【方案四：第一次评估时启用调试信息】
                    val_hit_ratio, val_mrr, val_ndcg = evaluate_full_ranking(
                        val_user_emb, val_item_emb, val_interactions, train_interactions, 
                        num_items_val, K=20, device=device, debug=(epoch == 0)  # 仅第一轮输出调试信息
                    )
                    
                    print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {loss.item():.4f} | "
                          f"Val Hit@20: {val_hit_ratio:.4f} | Val MRR: {val_mrr:.4f} | Val NDCG@20: {val_ndcg:.4f}")
                else:
                    print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {loss.item():.4f}")
        else:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {loss.item():.4f}")
    
    print("=" * 80)
    print("训练完成！开始测试...")

    # 测试
    # model.eval()
    # with torch.no_grad():
    #     test_x = data.x.to(device)
    #     test_edge_index = data.edge_index.to(device)
    #     test_embeddings = model(test_x, test_edge_index)
    #     # 计算测试集指标
    #     recall = recall_at_k(y_true=test_x, k=10)
    #     ndcg = ndcg_score()
    #     print(f"Test Recall@{args.k}: {recall:.4f}, NDCG@{args.k}: {ndcg:.4f}")
    # 测试
    model.eval()
    with torch.no_grad():

        test_x = torch.tensor(new_data['gb_features'], dtype=torch.float32).to(device)
        test_edge_index = torch.tensor(new_data['adj'], dtype=torch.long).to(device)
        test_embeddings = model(test_x, test_edge_index)  # 用粒球数据计算嵌入
        # L2归一化
        test_embeddings = F.normalize(test_embeddings, p=2, dim=1)
        
        # 修复：使用node_to_gb映射获取原始用户和物品嵌入（与训练逻辑一致）
        node_to_gb_test = new_data['node_to_gb']
        num_original_nodes_test = len(data.node_types)
        num_users_test = args.num_users
        num_items_test = num_original_nodes_test - num_users_test
        
        # 映射原始用户嵌入（多粒球均值聚合，与训练逻辑一致）
        test_user_emb = torch.zeros((num_users_test, args.embedding_dim), device=device)
        for user_id in range(num_users_test):
            gb_indices = node_to_gb_test.get(user_id, [])  # 获取用户所属的所有粒球索引（列表）
            if gb_indices:
                # 多粒球嵌入均值聚合
                test_user_emb[user_id] = torch.mean(test_embeddings[gb_indices], dim=0)
        
        # 映射原始物品嵌入（多粒球均值聚合，与训练逻辑一致）
        test_item_emb = torch.zeros((num_items_test, args.embedding_dim), device=device)
        for item_id in range(num_items_test):
            original_node_id = num_users_test + item_id
            gb_indices = node_to_gb_test.get(original_node_id, [])  # 获取物品所属的所有粒球索引（列表）
            if gb_indices:
                test_item_emb[item_id] = torch.mean(test_embeddings[gb_indices], dim=0)
        
        num_items = test_item_emb.shape[0]

        # 获取测试集交互和训练集交互
        test_interactions = data.interactions[data.test_mask]
        train_interactions = data.interactions[data.train_mask]
        if len(test_interactions) == 0:
            print("测试集为空，无法计算指标")
            return

        # 使用全物品排序评估，计算多个K值的指标
        print("\n测试集评估结果：")
        for K in [10, 20]:
            test_hit_ratio, test_mrr, test_ndcg = evaluate_full_ranking(
                test_user_emb, test_item_emb, test_interactions, train_interactions, 
                num_items, K=K, device=device
            )
            print(f"  K={K:2d} | Hit@{K}: {test_hit_ratio:.4f} | MRR: {test_mrr:.4f} | NDCG@{K}: {test_ndcg:.4f}")
if __name__ == '__main__':

    main()

