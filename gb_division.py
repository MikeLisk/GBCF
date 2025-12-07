from tools import *
import numpy as np
import networkx as nx
import matplotlib
from tools.add_noise import add_noise
matplotlib.use('TkAgg')
import scipy.sparse as sp
from tools.split_ball_purity import split_ball_purity
from tools.purification import purification
from tools.add_id import add_id
from tools.new_graph import new_graph
from tools.corse_split import initial_splite
from tools.add_purity import add_purity
from tools.split_ball_purity import split_ball_further



def gb_division(data, args):
    seed = 0

    #whether to add noise
    # if args.noise == 1:
    #     data = add_noise(data)
    C = []

    print(data)
    # data_file = data.x.numpy()
    total_balls_num = int(args.ball_r * len(data.node_types))
    # adjacency_matrix = sp.coo_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0], data.edge_index[1])),
    #                                  shape=(data.y.shape[0], data.y.shape[0]),
    #                                 #  dtype=np.float32)
    #构建邻接矩阵
    adjacency_matrix = sp.coo_matrix(
    (np.ones(data.edge_index.shape[1]), (data.edge_index[0].numpy(), data.edge_index[1].numpy())), dtype=np.float32
)
    # for i in range(0, len(data.test_mask)):
    #     if data.test_mask[i]:
    #         data.y[i] = -1

    # data_labels = data.node_types.numpy()
    node_types = data.node_types.numpy()
    
    # 【数据验证】记录粒球划分使用的边数量，确保只使用训练集边
    print(f"粒球划分使用的边数量: {data.edge_index.shape[1]}")
    print(f"粒球划分使用的节点数量: {len(data.node_types)}")
    
    #load Dataset
    graph = get_dataset.get_dataset(adjacency_matrix, node_types, seed)

    # 【修复】推荐任务中不删除孤立节点
    # 原因：使用训练集边构建图后，只在验证/测试集中出现的节点会变成孤立节点
    # 但我们需要保留所有用户和物品节点，以便在评估时能对它们进行预测
    # graph = del_outlier.del_outlier(graph)  # 注释掉孤立节点删除
    print(f"保留所有节点（包括孤立节点），节点总数: {graph.number_of_nodes()}")

    # node_attributes = nx.get_node_attributes(graph, "attributes")
    # attributes = np.array(list(node_attributes.values()))
    # node_labels = nx.get_node_attributes(graph, "label")
    # labels = np.array(list(node_labels.values()))
    # labels = np.expand_dims(labels, axis=0)
    # labels = np.reshape(labels, (nx.number_of_nodes(graph), 1))


    total_degree_dict_old = dict(graph.degree())
    total_degree_dict = {}
    id = 0
    for key, value in total_degree_dict_old.items():
        total_degree_dict[id] = value
        id += 1

    id_dict = {}
    id_dict_oldtonew = {}
    for new, old in enumerate(total_degree_dict_old):
        id_dict[new] = old
        id_dict_oldtonew[old] = new

    indices = []
    for index in total_degree_dict_old:
        indices.append(index)
    indices = np.expand_dims(indices, axis=0)
    indices = np.reshape(indices, (nx.number_of_nodes(graph), 1))
    data_np = indices
    # 添加虚拟标签列（全0），以便 corse_split 正确分组
    dummy_labels = np.zeros((len(indices), 1))
    data_np = np.hstack((data_np, dummy_labels))
    data_np = add_id(data_np)

    C.append([data_np, total_degree_dict])


    # coarse division of granules
    new_C = initial_splite(C, graph, id_dict, id_dict_oldtonew, total_degree_dict)


    target = 1
    while len(new_C) > total_balls_num:
        cut_pos = 0
        new_C.sort(key=lambda x: len(x[0]))
        for i in range(0, len(new_C)):
            if len(new_C[i][0]) == target+1:
                cut_pos = i
                break
        target += 1
        new_C = new_C[cut_pos:]

    new_C = add_purity(new_C)


    #binary division of granules
    new_C = split_ball_purity(graph, id_dict, new_C, total_degree_dict, total_balls_num)

    if len(new_C) < total_balls_num:
        new_C = split_ball_further(graph, id_dict, new_C, total_degree_dict, total_balls_num)

    new_C = purification(new_C)

    # 【修复】对于无属性节点（推荐任务），不需要计算粒球的聚合特征
    # 直接使用全0或随机特征作为占位符（实际使用RecGCN的可学习嵌入）
    print("\n【粒球特征构造】由于节点无属性，跳过特征聚合，使用占位符特征。")
    # 生成占位符特征，形状为 (粒球数, 1)，避免空值报错
    GB_features = np.zeros((len(new_C), 1))
    
    # 统计粒球类型分布（仅用于展示）
    mixed_balls = 0
    user_only_balls = 0
    item_only_balls = 0
    num_users = (data.node_types == 0).sum().item()
    
    for GB in new_C:
        gb_data = np.array(GB[0])
        node_ids = gb_data[:, -1].astype(int)
        user_mask = node_ids < num_users
        has_users = np.any(user_mask)
        has_items = np.any(~user_mask)
        
        if has_users and has_items:
            mixed_balls += 1
        elif has_users:
            user_only_balls += 1
        else:
            item_only_balls += 1
            
    total_balls = len(new_C)
    print(f"  总粒球数: {total_balls}")
    print(f"  混合粒球: {mixed_balls} ({100*mixed_balls/total_balls:.1f}%)")
    print(f"  仅用户: {user_only_balls} ({100*user_only_balls/total_balls:.1f}%)")
    print(f"  仅物品: {item_only_balls} ({100*item_only_balls/total_balls:.1f}%)")
    print()


    #build granules
    GB_graph = new_graph(new_C, graph)
    # 记录原始节点到粒球的映射（原始节点ID -> 粒球索引）
    node_to_gb = {}
    for gb_idx, gb in enumerate(new_C):
        data = np.array(gb[0])
        node_ids = data[:, -1].astype(int)  # 原始节点ID（在粒球中的节点）
        for node_id in node_ids:
            if node_id not in node_to_gb:
                node_to_gb[node_id] = []
            node_to_gb[node_id].append(gb_idx) # 一个节点可能属于多个粒球，这里取最后一个（或根据需求调整）
    new_f = {}
    gb_labels = []
    for GB in new_C:
        gb_labels.append(GB[-1])
    new_f['gb_labels'] = np.array(gb_labels)
    C_adj = sp.coo_matrix(nx.to_numpy_array(GB_graph))
    C_adj = np.vstack((C_adj.row, C_adj.col))
    new_f['adj'] = C_adj
    new_f['gb_features'] = np.array(GB_features)
    new_f['node_to_gb'] = node_to_gb 
    return new_f





