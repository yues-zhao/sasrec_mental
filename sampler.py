import numpy as np
import json
from multiprocessing import Process, Queue


def random_neq(l, r, s):
    """Random sample a number in [l, r) that is not in set s"""
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def neg_sample_from_cat(cat_item_map, cat_id, pos_item, user_items):
    """
    Negative sampling from the same category
    Returns a negative item from the same category that is not the positive item
    and not in the user's interaction history
    """
    items = cat_item_map.get(str(cat_id), [])
    if len(items) <= 1:
        return -1  # Cannot sample from single-item category
    
    neg = np.random.choice(items)
    exclude_set = user_items | {pos_item}
    attempts = 0
    while neg in exclude_set and attempts < 100:
        neg = np.random.choice(items)
        attempts += 1
    
    if neg in exclude_set:
        return -1  # Failed to find negative sample
    return neg


def neg_sample_global(all_item_list, pos_item, user_items):
    """
    Global negative sampling for single-item categories
    Returns a negative item from all items that is not the positive item
    and not in the user's interaction history
    """
    neg = np.random.choice(all_item_list)
    exclude_set = user_items | {pos_item}
    attempts = 0
    while neg in exclude_set and attempts < 100:
        neg = np.random.choice(all_item_list)
        attempts += 1
    
    if neg in exclude_set:
        return -1  # Failed to find negative sample
    return neg


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED,
                    use_features=False, user_train_features=None, 
                    cat_item_map=None, all_item_list=None):
    """Sample function for generating training batches"""
    
    # Load sampling data if provided
    if cat_item_map is None:
        cat_item_map = {}
    if all_item_list is None:
        all_item_list = list(range(1, itemnum + 1))
    
    # Build user item sets for faster exclusion
    user_item_sets = {}
    for user in user_train:
        user_item_sets[user] = set(user_train[user])
    
    # Build category lookup from cat_item_map
    item_to_cat = {}
    for cat_id, items in cat_item_map.items():
        for item in items:
            item_to_cat[int(item)] = int(cat_id)
    
    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1:
            user = np.random.randint(1, usernum + 1)
        
        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        
        # Feature arrays
        if use_features:
            price_seq = np.zeros([maxlen], dtype=np.float32)
            price_pos = np.zeros([maxlen], dtype=np.float32)
            price_neg = np.zeros([maxlen], dtype=np.float32)
            
            cat_seq = np.zeros([maxlen], dtype=np.int32)
            cat_pos = np.zeros([maxlen], dtype=np.int32)
            cat_neg = np.zeros([maxlen], dtype=np.int32)
            
            price_dev_seq = np.zeros([maxlen], dtype=np.float32)
            price_dev_pos = np.zeros([maxlen], dtype=np.float32)
            price_dev_neg = np.zeros([maxlen], dtype=np.float32)
            
            time_dev_seq = np.zeros([maxlen], dtype=np.float32)
            time_dev_pos = np.zeros([maxlen], dtype=np.float32)
            time_dev_neg = np.zeros([maxlen], dtype=np.float32)
            
            features = user_train_features.get(user, [])
            feat_dict = {f['item']: f for f in features}
        
        nxt = user_train[user][-1]
        idx = maxlen - 1
        
        ts = user_item_sets[user]
        
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            
            if nxt != 0:
                # Get category of positive item
                pos_cat = item_to_cat.get(nxt, -1)
                cat_items_count = len(cat_item_map.get(str(pos_cat), []))
                
                # Mixed negative sampling strategy
                if cat_items_count > 1:
                    # Category-based negative sampling
                    neg_item = neg_sample_from_cat(cat_item_map, pos_cat, nxt, ts)
                    if neg_item == -1:
                        neg_item = neg_sample_global(all_item_list, nxt, ts)
                else:
                    # Global negative sampling for single-item categories
                    neg_item = neg_sample_global(all_item_list, nxt, ts)
                
                if neg_item == -1:
                    neg_item = random_neq(1, itemnum + 1, ts)
                
                neg[idx] = neg_item
                
                # Fill feature arrays
                if use_features:
                    pos_feat = feat_dict.get(nxt, {})
                    neg_feat = feat_dict.get(neg_item, {})
                    
                    price_pos[idx] = pos_feat.get('price', 0.0)
                    cat_pos[idx] = pos_feat.get('category', 0)
                    price_dev_pos[idx] = pos_feat.get('price_dev', 0.0)
                    time_dev_pos[idx] = pos_feat.get('time_dev', 0.0)
                    
                    price_neg[idx] = neg_feat.get('price', 0.0)
                    cat_neg[idx] = neg_feat.get('category', 0)
                    price_dev_neg[idx] = neg_feat.get('price_dev', 0.0)
                    time_dev_neg[idx] = neg_feat.get('time_dev', 0.0)
            
            nxt = i
            idx -= 1
            if idx == -1:
                break
        
        if use_features:
            # Fill sequence features (shifted by 1)
            for j in range(maxlen):
                item_id = seq[j]
                if item_id != 0:
                    seq_feat = feat_dict.get(item_id, {})
                    price_seq[j] = seq_feat.get('price', 0.0)
                    cat_seq[j] = seq_feat.get('category', 0)
                    price_dev_seq[j] = seq_feat.get('price_dev', 0.0)
                    time_dev_seq[j] = seq_feat.get('time_dev', 0.0)
            
            return (user, seq, pos, neg,
                    price_seq, price_pos, price_neg,
                    cat_seq, cat_pos, cat_neg,
                    price_dev_seq, price_dev_pos, price_dev_neg,
                    time_dev_seq, time_dev_pos, time_dev_neg)
        else:
            return (user, seq, pos, neg)
    
    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())
        
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1,
                 use_features=False, user_train_features=None, cat_item_map=None, all_item_list=None):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        
        # Load sampling data
        if cat_item_map is None:
            cat_item_map = {}
        if all_item_list is None:
            all_item_list = list(range(1, itemnum + 1))
        
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9),
                                                      use_features,
                                                      user_train_features,
                                                      cat_item_map,
                                                      all_item_list
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()
    
    def next_batch(self):
        return self.result_queue.get()
    
    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
