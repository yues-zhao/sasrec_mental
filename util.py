import sys
import copy
import random
import numpy as np
import json
from collections import defaultdict


def data_partition(fname, use_features=False):
    """Load and partition data into train/valid/test sets"""
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    
    if use_features:
        # 加载带多特征的数据
        User_features = defaultdict(list)
        # 构建全局 item_id -> features 映射（用于评估时获取候选物品特征）
        item_features = {}
        
        with open('data/%s.txt' % fname, 'r') as f:
            for line in f:
                parts = line.rstrip().split(' ')
                if len(parts) == 6:
                    u, i, price, cat, price_dev, time_dev = parts
                    u, i, cat = int(u), int(i), int(cat)
                    price, price_dev, time_dev = float(price), float(price_dev), float(time_dev)
                    
                    usernum = max(u, usernum)
                    itemnum = max(i, itemnum)
                    User[u].append(i)
                    User_features[u].append({
                        'item': i,
                        'price': price,
                        'category': cat,
                        'price_dev': price_dev,
                        'time_dev': time_dev
                    })
                    # 记录每个 item 的特征（去重，同一 item 的特征应该相同）
                    if i not in item_features:
                        item_features[i] = {
                            'price': price,
                            'category': cat,
                            'price_dev': price_dev,
                            'time_dev': time_dev
                        }
        
        for user in User:
            nfeedback = len(User[user])
            if nfeedback < 3:
                user_train[user] = User[user]
                user_valid[user] = []
                user_test[user] = []
            else:
                user_train[user] = User[user][:-2]
                user_valid[user] = []
                user_valid[user].append(User[user][-2])
                user_test[user] = []
                user_test[user].append(User[user][-1])
        
        user_train_features = {}
        user_valid_features = {}
        user_test_features = {}
        
        for user in User_features:
            features = User_features[user]
            nfeedback = len(features)
            if nfeedback < 3:
                user_train_features[user] = features
                user_valid_features[user] = []
                user_test_features[user] = []
            else:
                user_train_features[user] = features[:-2]
                user_valid_features[user] = [features[-2]]
                user_test_features[user] = [features[-1]]
        
        return [user_train, user_valid, user_test, usernum, itemnum,
                user_train_features, user_valid_features, user_test_features,
                item_features]
    else:
        # 原始数据格式（向后兼容）
        with open('data/%s.txt' % fname, 'r') as f:
            for line in f:
                u, i = line.rstrip().split(' ')
                u = int(u)
                i = int(i)
                usernum = max(u, usernum)
                itemnum = max(i, itemnum)
                User[u].append(i)

        for user in User:
            nfeedback = len(User[user])
            if nfeedback < 3:
                user_train[user] = User[user]
                user_valid[user] = []
                user_test[user] = []
            else:
                user_train[user] = User[user][:-2]
                user_valid[user] = []
                user_valid[user].append(User[user][-2])
                user_test[user] = []
                user_test[user].append(User[user][-1])
        
        return [user_train, user_valid, user_test, usernum, itemnum]


def evaluate(model, dataset, args, device):
    """Evaluate model on test set"""
    # Handle both 5-value (SASRec) and 9-value (SASRecMoE with item_features) datasets
    if len(dataset) == 9:
        train, valid, test, usernum, itemnum, _, _, _, item_features = copy.deepcopy(dataset)
        use_moe = True
    elif len(dataset) == 8:
        train, valid, test, usernum, itemnum, _, _, _ = copy.deepcopy(dataset)
        use_moe = False
    else:
        train, valid, test, usernum, itemnum = copy.deepcopy(dataset)
        use_moe = False

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    
    model.eval()
    with torch.no_grad():
        for u in users:
            if len(train[u]) < 1 or len(test[u]) < 1:
                continue

            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            seq[idx] = valid[u][0]
            idx -= 1
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1:
                    break
            
            rated = set(train[u])
            rated.add(0)
            item_idx = [test[u][0]]
            for _ in range(100):
                t = np.random.randint(1, itemnum + 1)
                while t in rated:
                    t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)

            # Convert to tensors (使用non_blocking加速传输)
            seq_tensor = torch.LongTensor(seq).unsqueeze(0).to(device, non_blocking=True)
            item_idx_tensor = torch.LongTensor(item_idx).to(device, non_blocking=True)

            # 为MoE模型准备候选物品特征
            if use_moe and hasattr(model, 'price_tower'):
                # 获取序列特征
                seq_prices = np.zeros([args.maxlen], dtype=np.float32)
                seq_cats = np.zeros([args.maxlen], dtype=np.int32)
                seq_price_devs = np.zeros([args.maxlen], dtype=np.float32)
                seq_time_devs = np.zeros([args.maxlen], dtype=np.float32)

                for j in range(args.maxlen):
                    item_id = seq[j]
                    if item_id != 0 and item_id in item_features:
                        feat = item_features[item_id]
                        seq_prices[j] = feat['price']
                        seq_cats[j] = feat['category']
                        seq_price_devs[j] = feat['price_dev']
                        seq_time_devs[j] = feat['time_dev']

                # 获取候选物品特征
                item_prices = np.zeros([len(item_idx)], dtype=np.float32)
                item_cats = np.zeros([len(item_idx)], dtype=np.int32)
                item_price_devs = np.zeros([len(item_idx)], dtype=np.float32)
                item_time_devs = np.zeros([len(item_idx)], dtype=np.float32)

                for j, item_id in enumerate(item_idx):
                    if item_id in item_features:
                        feat = item_features[item_id]
                        item_prices[j] = feat['price']
                        item_cats[j] = feat['category']
                        item_price_devs[j] = feat['price_dev']
                        item_time_devs[j] = feat['time_dev']

                price_seq_tensor = torch.FloatTensor(seq_prices).unsqueeze(0).to(device, non_blocking=True)
                cat_seq_tensor = torch.LongTensor(seq_cats).unsqueeze(0).to(device, non_blocking=True)
                price_dev_seq_tensor = torch.FloatTensor(seq_price_devs).unsqueeze(0).to(device, non_blocking=True)
                time_dev_seq_tensor = torch.FloatTensor(seq_time_devs).unsqueeze(0).to(device, non_blocking=True)

                price_item_tensor = torch.FloatTensor(item_prices).to(device, non_blocking=True)
                cat_item_tensor = torch.LongTensor(item_cats).to(device, non_blocking=True)
                price_dev_item_tensor = torch.FloatTensor(item_price_devs).to(device, non_blocking=True)
                time_dev_item_tensor = torch.FloatTensor(item_time_devs).to(device, non_blocking=True)

                predictions = -model.predict(
                    seq_tensor, item_idx_tensor,
                    price_seq=price_seq_tensor, cat_seq=cat_seq_tensor,
                    time_dev_seq=time_dev_seq_tensor, price_dev_seq=price_dev_seq_tensor,
                    item_prices=price_item_tensor, item_cats=cat_item_tensor,
                    item_price_devs=price_dev_item_tensor, item_time_devs=time_dev_item_tensor
                ).cpu().numpy()
            else:
                predictions = -model.predict(seq_tensor, item_idx_tensor).cpu().numpy()
            
            rank = predictions.argsort().argsort()[0]

            valid_user += 1

            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1
            
            if valid_user % 100 == 0:
                print('.', end='')
                sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args, device):
    """Evaluate model on validation set"""
    # Handle both 5-value (SASRec) and 9-value (SASRecMoE with item_features) datasets
    if len(dataset) == 9:
        train, valid, test, usernum, itemnum, _, _, _, item_features = copy.deepcopy(dataset)
        use_moe = True
    elif len(dataset) == 8:
        train, valid, test, usernum, itemnum, _, _, _ = copy.deepcopy(dataset)
        use_moe = False
    else:
        train, valid, test, usernum, itemnum = copy.deepcopy(dataset)
        use_moe = False

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    
    model.eval()
    with torch.no_grad():
        for u in users:
            if len(train[u]) < 1 or len(valid[u]) < 1:
                continue

            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1:
                    break

            rated = set(train[u])
            rated.add(0)
            item_idx = [valid[u][0]]
            for _ in range(100):
                t = np.random.randint(1, itemnum + 1)
                while t in rated:
                    t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)

            # Convert to tensors (使用non_blocking加速传输)
            seq_tensor = torch.LongTensor(seq).unsqueeze(0).to(device, non_blocking=True)
            item_idx_tensor = torch.LongTensor(item_idx).to(device, non_blocking=True)

            # 为MoE模型准备候选物品特征
            if use_moe and hasattr(model, 'price_tower'):
                # 获取序列特征
                seq_prices = np.zeros([args.maxlen], dtype=np.float32)
                seq_cats = np.zeros([args.maxlen], dtype=np.int32)
                seq_price_devs = np.zeros([args.maxlen], dtype=np.float32)
                seq_time_devs = np.zeros([args.maxlen], dtype=np.float32)

                for j in range(args.maxlen):
                    item_id = seq[j]
                    if item_id != 0 and item_id in item_features:
                        feat = item_features[item_id]
                        seq_prices[j] = feat['price']
                        seq_cats[j] = feat['category']
                        seq_price_devs[j] = feat['price_dev']
                        seq_time_devs[j] = feat['time_dev']

                # 获取候选物品特征
                item_prices = np.zeros([len(item_idx)], dtype=np.float32)
                item_cats = np.zeros([len(item_idx)], dtype=np.int32)
                item_price_devs = np.zeros([len(item_idx)], dtype=np.float32)
                item_time_devs = np.zeros([len(item_idx)], dtype=np.float32)

                for j, item_id in enumerate(item_idx):
                    if item_id in item_features:
                        feat = item_features[item_id]
                        item_prices[j] = feat['price']
                        item_cats[j] = feat['category']
                        item_price_devs[j] = feat['price_dev']
                        item_time_devs[j] = feat['time_dev']

                price_seq_tensor = torch.FloatTensor(seq_prices).unsqueeze(0).to(device, non_blocking=True)
                cat_seq_tensor = torch.LongTensor(seq_cats).unsqueeze(0).to(device, non_blocking=True)
                price_dev_seq_tensor = torch.FloatTensor(seq_price_devs).unsqueeze(0).to(device, non_blocking=True)
                time_dev_seq_tensor = torch.FloatTensor(seq_time_devs).unsqueeze(0).to(device, non_blocking=True)

                price_item_tensor = torch.FloatTensor(item_prices).to(device, non_blocking=True)
                cat_item_tensor = torch.LongTensor(item_cats).to(device, non_blocking=True)
                price_dev_item_tensor = torch.FloatTensor(item_price_devs).to(device, non_blocking=True)
                time_dev_item_tensor = torch.FloatTensor(item_time_devs).to(device, non_blocking=True)

                predictions = -model.predict(
                    seq_tensor, item_idx_tensor,
                    price_seq=price_seq_tensor, cat_seq=cat_seq_tensor,
                    time_dev_seq=time_dev_seq_tensor, price_dev_seq=price_dev_seq_tensor,
                    item_prices=price_item_tensor, item_cats=cat_item_tensor,
                    item_price_devs=price_dev_item_tensor, item_time_devs=time_dev_item_tensor
                ).cpu().numpy()
            else:
                predictions = -model.predict(seq_tensor, item_idx_tensor).cpu().numpy()
            
            rank = predictions.argsort().argsort()[0]

            valid_user += 1

            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1
            
            if valid_user % 100 == 0:
                print('.', end='')
                sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


import torch
