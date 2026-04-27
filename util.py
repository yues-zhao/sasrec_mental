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
                user_train_features, user_valid_features, user_test_features]
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
    # Handle both 5-value (SASRec) and 8-value (SASRecMoE) datasets
    if len(dataset) == 8:
        train, valid, test, usernum, itemnum, _, _, _ = copy.deepcopy(dataset)
    else:
        train, valid, test, usernum, itemnum = copy.deepcopy(dataset)

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

            # Convert to tensors
            seq_tensor = torch.LongTensor(seq).unsqueeze(0).to(device)
            item_idx_tensor = torch.LongTensor(item_idx).to(device)
            
            # Get predictions
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
    # Handle both 5-value (SASRec) and 8-value (SASRecMoE) datasets
    if len(dataset) == 8:
        train, valid, test, usernum, itemnum, _, _, _ = copy.deepcopy(dataset)
    else:
        train, valid, test, usernum, itemnum = copy.deepcopy(dataset)

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

            # Convert to tensors
            seq_tensor = torch.LongTensor(seq).unsqueeze(0).to(device)
            item_idx_tensor = torch.LongTensor(item_idx).to(device)
            
            # Get predictions
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
