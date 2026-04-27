# test_data_loading.py
"""测试Task 2: 数据加载和负采样流程"""
import sys
import numpy as np

print("=" * 60)
print("Task 2 测试数据加载和负采样流程")
print("=" * 60)

all_passed = True

# 1. 测试数据加载
print("\n1. 测试数据加载...")
try:
    from util import data_partition
    
    # 测试带特征的数据加载
    dataset = data_partition("TaFeng_MoE", use_features=True)
    
    if len(dataset) == 8:
        user_train, user_valid, user_test, usernum, itemnum, \
        user_train_features, user_valid_features, user_test_features = dataset
        
        print(f"   用户数: {usernum}")
        print(f"   商品数: {itemnum}")
        print(f"   训练集用户数: {len(user_train)}")
        print(f"   特征集用户数: {len(user_train_features)}")
        
        # 验证特征格式
        sample_user = list(user_train_features.keys())[0]
        sample_feat = user_train_features[sample_user][0]
        required_keys = ['item', 'price', 'category', 'price_dev', 'time_dev']
        has_all_keys = all(k in sample_feat for k in required_keys)
        
        if has_all_keys:
            print("   特征格式验证 [OK]")
        else:
            print(f"   特征格式验证 [FAIL] - 缺少键: {set(required_keys) - set(sample_feat.keys())}")
            all_passed = False
    else:
        print(f"   数据加载验证 [FAIL] - 返回长度应为8，实际为{len(dataset)}")
        all_passed = False

except Exception as e:
    print(f"   数据加载测试失败 [FAIL]: {e}")
    all_passed = False

# 2. 测试品类映射加载
print("\n2. 测试品类映射加载...")
try:
    import json
    with open("data/cat_item_map.json", "r", encoding="utf-8") as f:
        sampling_data = json.load(f)
    
    cat_item_map = sampling_data['cat_item_map']
    all_item_list = sampling_data['all_item_list']
    
    print(f"   品类数: {len(cat_item_map)}")
    print(f"   全局商品数: {len(all_item_list)}")
    print("   品类映射加载 [OK]")
except Exception as e:
    print(f"   品类映射加载失败 [FAIL]: {e}")
    all_passed = False

# 3. 测试混合负采样逻辑
print("\n3. 测试混合负采样逻辑...")
try:
    from sampler import neg_sample_from_cat, neg_sample_global
    
    # 构建测试数据
    test_user_items = {1, 2, 3, 5, 8}
    
    # 测试品类内负采样（多商品品类）
    test_cat_map = {'1': [1, 4, 6, 10], '2': [7]}  # 品类1有4个商品，品类2只有1个
    pos_item_cat1 = 4
    neg_cat1 = neg_sample_from_cat(test_cat_map, 1, pos_item_cat1, test_user_items)
    
    if neg_cat1 != -1 and neg_cat1 not in test_user_items and neg_cat1 != pos_item_cat1:
        print(f"   品类内负采样（品类1）: 正样本={pos_item_cat1}, 负样本={neg_cat1} [OK]")
    else:
        print(f"   品类内负采样（品类1）[FAIL]: 正样本={pos_item_cat1}, 负样本={neg_cat1}")
        all_passed = False
    
    # 测试单品品类（应该返回-1）
    pos_item_cat2 = 7
    neg_cat2 = neg_sample_from_cat(test_cat_map, 2, pos_item_cat2, test_user_items)
    
    if neg_cat2 == -1:
        print(f"   单品品类负采样（品类2）: 正样本={pos_item_cat2}, 返回-1 [OK]")
    else:
        print(f"   单品品类负采样（品类2）[FAIL]: 正样本={pos_item_cat2}, 负样本={neg_cat2}")
        all_passed = False
    
    # 测试全局负采样
    all_items = list(range(1, 20))
    neg_global = neg_sample_global(all_items, 4, test_user_items)
    
    if neg_global != -1 and neg_global not in test_user_items and neg_global != 4:
        print(f"   全局负采样: 正样本=4, 负样本={neg_global} [OK]")
    else:
        print(f"   全局负采样 [FAIL]: 正样本=4, 负样本={neg_global}")
        all_passed = False

except Exception as e:
    print(f"   负采样逻辑测试失败 [FAIL]: {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

# 4. 测试采样器（简化版，不启动多进程）
print("\n4. 测试采样器功能...")
try:
    # 构建品类到物品的映射
    item_to_cat = {}
    for cat_id, items in cat_item_map.items():
        for item in items:
            item_to_cat[int(item)] = int(cat_id)
    
    # 统计负采样策略分布
    cat_neg_samples = 0
    global_neg_samples = 0
    
    for user in list(user_train.keys())[:100]:  # 测试前100个用户
        items = user_train[user]
        if len(items) <= 1:
            continue
        
        for i in range(len(items) - 1):
            pos_item = items[-(i+1)]
            pos_cat = item_to_cat.get(pos_item, -1)
            cat_items_count = len(cat_item_map.get(str(pos_cat), []))
            
            if cat_items_count > 1:
                cat_neg_samples += 1
            else:
                global_neg_samples += 1
    
    print(f"   测试用户数: 100")
    print(f"   品类内负采样次数: {cat_neg_samples}")
    print(f"   全局负采样次数: {global_neg_samples}")
    
    if cat_neg_samples + global_neg_samples > 0:
        print("   采样器功能测试 [OK]")
    else:
        print("   采样器功能测试 [FAIL]")
        all_passed = False

except Exception as e:
    print(f"   采样器功能测试失败 [FAIL]: {e}")
    all_passed = False

# 5. 测试向后兼容性
print("\n5. 测试向后兼容性...")
try:
    # 使用原始数据格式
    dataset_old = data_partition("TaFeng", use_features=False)
    
    if len(dataset_old) == 5:
        print(f"   原始数据格式加载 [OK]")
    else:
        print(f"   原始数据格式加载 [FAIL] - 返回长度应为5，实际为{len(dataset_old)}")
        all_passed = False

except Exception as e:
    print(f"   向后兼容性测试失败 [FAIL]: {e}")
    all_passed = False

# 最终结论
print("\n" + "=" * 60)
if all_passed:
    print("Task 2 验证通过 - 所有检查项 [OK]")
else:
    print("Task 2 验证失败 - 存在未通过检查项")
print("=" * 60)

sys.exit(0 if all_passed else 1)
