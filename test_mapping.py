import torch
from gb_division import gb_division
from split_test import load_recommendation_dataset
import argparse

# 设置参数
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="sparse_yelp")
parser.add_argument('--ball_r', type=float, default=0.3)
parser.add_argument('--noise', type=int, default=0)
parser.add_argument('--embedding_dim', type=int, default=128)
args = parser.parse_args([])

# 加载数据
print("加载数据...")
data = load_recommendation_dataset(name=args.dataset)
args.num_features = data.x.shape[1]
args.num_users = (data.node_types == 0).sum().item()

print(f"用户数: {args.num_users}")
print(f"物品数: {data.x.shape[0] - args.num_users}")
print(f"节点总数: {data.x.shape[0]}")

# 粒球划分
print("\n执行粒球划分...")
new_data = gb_division(data, args)

# 检查映射
node_to_gb = new_data['node_to_gb']
print(f"\nnode_to_gb包含的节点数: {len(node_to_gb)}")

# 统计有多少用户和物品有映射
user_mapped = 0
item_mapped = 0
num_items = data.x.shape[0] - args.num_users

for user_id in range(args.num_users):
    if node_to_gb.get(user_id, []):
        user_mapped += 1

for item_id in range(num_items):
    original_node_id = args.num_users + item_id
    if node_to_gb.get(original_node_id, []):
        item_mapped += 1

print(f"\n映射统计：")
print(f"  用户映射覆盖率: {user_mapped}/{args.num_users} ({100*user_mapped/args.num_users:.2f}%)")
print(f"  物品映射覆盖率: {item_mapped}/{num_items} ({100*item_mapped/num_items:.2f}%)")

# 检查验证集用户的映射
val_interactions = data.interactions[data.val_mask]
val_user_ids = val_interactions[:, 0].unique().numpy()
val_users_mapped = sum(1 for uid in val_user_ids if node_to_gb.get(uid, []))
print(f"  验证集用户映射: {val_users_mapped}/{len(val_user_ids)} ({100*val_users_mapped/len(val_user_ids):.2f}%)")

# 检查一些具体的样本
print(f"\n检查前5个用户的映射：")
for user_id in range(min(5, args.num_users)):
    gb_indices = node_to_gb.get(user_id, [])
    print(f"  用户{user_id}: 粒球{gb_indices}")

print(f"\n检查前5个物品的映射：")
for item_id in range(min(5, num_items)):
    original_node_id = args.num_users + item_id
    gb_indices = node_to_gb.get(original_node_id, [])
    print(f"  物品{item_id} (节点{original_node_id}): 粒球{gb_indices}")
