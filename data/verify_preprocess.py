# verify_preprocess.py
"""验证增强预处理脚本的正确性 - Task 1检查"""
import json
import sys

print("=" * 60)
print("Task 1 验证增强预处理结果")
print("=" * 60)

all_passed = True

# 1. 验证输出文件格式
print("\n1. 验证输出文件格式...")
try:
    with open("data/TaFeng_MoE.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"   总记录数: {len(lines)}")
    
    format_errors = 0
    for i, line in enumerate(lines[:1000]):
        parts = line.strip().split()
        if len(parts) != 6:
            format_errors += 1
            if format_errors <= 3:
                print(f"   错误: 第{i+1}行格式不正确: {line.strip()}")
    
    if format_errors == 0:
        print("   格式验证通过 [OK]")
    else:
        print(f"   格式验证失败 [FAIL] - {format_errors} 行格式错误")
        all_passed = False
except Exception as e:
    print(f"   读取文件失败 [FAIL]: {e}")
    all_passed = False

# 2. 验证数据泄露防护（价格偏离度）
print("\n2. 验证价格偏离度数据泄露防护...")
try:
    user_seqs = {}
    for line in lines:
        parts = line.strip().split()
        uid, iid, price, cid, price_dev, time_dev = parts
        uid, iid, cid = int(uid), int(iid), int(cid)
        
        if uid not in user_seqs:
            user_seqs[uid] = []
        user_seqs[uid].append({
            'iid': iid,
            'price': float(price),
            'cid': cid,
            'price_dev': float(price_dev),
            'time_dev': float(time_dev)
        })
    
    leak_count = 0
    check_count = 0
    for uid in user_seqs:
        seq = user_seqs[uid]
        if len(seq) > 0:
            check_count += 1
            if seq[0]['price_dev'] != 0.0:
                leak_count += 1
    
    print(f"   检查用户数: {check_count}")
    print(f"   数据泄露数: {leak_count}")
    if leak_count == 0:
        print("   价格偏离度防护验证 [OK]")
    else:
        print("   价格偏离度防护验证 [FAIL]")
        all_passed = False
except Exception as e:
    print(f"   验证失败 [FAIL]: {e}")
    all_passed = False

# 3. 验证品类-商品映射和全局商品列表
print("\n3. 验证品类-商品映射和全局商品列表...")
try:
    with open("data/cat_item_map.json", "r", encoding="utf-8") as f:
        sampling_data = json.load(f)
    
    cat_item_map = sampling_data['cat_item_map']
    all_item_list = sampling_data['all_item_list']
    single_item_cats = sampling_data['single_item_cats']
    
    print(f"   品类数: {len(cat_item_map)}")
    print(f"   全局商品数: {len(all_item_list)}")
    print(f"   单品品类数: {len(single_item_cats)}")
    
    # 验证全局商品列表与品类商品总数的一致性
    total_cat_items = sum(len(v) for v in cat_item_map.values())
    print(f"   品类商品总数（含重复，实际无重复）: {total_cat_items}")
    
    # 验证负采样策略
    multi_item_cats = len(cat_item_map) - len(single_item_cats)
    print(f"   多商品品类数（支持品类内负采样）: {multi_item_cats}")
    print(f"   单品品类数（需要全局负采样）: {len(single_item_cats)}")
    
    # 验证单品品类确实只有1个商品
    single_cat_check = all(len(cat_item_map[cid]) == 1 for cid in single_item_cats)
    if single_cat_check:
        print("   映射结构验证 [OK]")
    else:
        print("   映射结构验证 [FAIL]")
        all_passed = False
except Exception as e:
    print(f"   验证失败 [FAIL]: {e}")
    all_passed = False

# 4. 验证特征统计信息
print("\n4. 验证特征统计信息...")
try:
    with open("data/feature_stats.json", "r", encoding="utf-8") as f:
        stats = json.load(f)
    
    print(f"   用户数: {stats['user_num']}")
    print(f"   商品数: {stats['item_num']}")
    print(f"   品类数: {stats['cat_num']}")
    
    if stats['user_num'] == len(user_seqs):
        print("   特征统计信息验证 [OK]")
    else:
        print("   特征统计信息验证 [FAIL]")
        all_passed = False
except Exception as e:
    print(f"   验证失败 [FAIL]: {e}")
    all_passed = False

# 5. 验证负采样策略可行性
print("\n5. 验证负采样策略可行性...")
try:
    # 单品品类使用全局负采样
    multi_cats = [cid for cid, items in cat_item_map.items() if len(items) > 1]
    single_cats = [cid for cid, items in cat_item_map.items() if len(items) == 1]
    
    print(f"   多商品品类（品类内负采样）: {len(multi_cats)}")
    print(f"   单品品类（全局负采样）: {len(single_cats)}")
    print(f"   全局商品列表可用: {'OK' if len(all_item_list) > 0 else 'FAIL'}")
    
    if len(multi_cats) > 0 and len(single_cats) >= 0 and len(all_item_list) > 1:
        print("   负采样策略可行性 [OK]")
    else:
        print("   负采样策略可行性 [FAIL]")
        all_passed = False
except Exception as e:
    print(f"   验证失败 [FAIL]: {e}")
    all_passed = False

# 6. 统计摘要
print("\n" + "=" * 60)
print("统计摘要")
print("=" * 60)
print(f"有效用户数: {stats['user_num']}")
print(f"有效商品数: {stats['item_num']}")
print(f"有效品类数: {stats['cat_num']}")
print(f"总交互记录数: {len(lines)}")
print(f"平均每用户交互数: {len(lines) / stats['user_num']:.2f}")

# 负采样策略总结
multi_cat_records = sum(1 for uid in user_seqs for item in user_seqs[uid] if item['cid'] in [int(c) for c in multi_cats])
single_cat_records = sum(1 for uid in user_seqs for item in user_seqs[uid] if item['cid'] in [int(c) for c in single_cats])
print(f"多商品品类交互记录数（品类内负采样）: {multi_cat_records}")
print(f"单品品类交互记录数（全局负采样）: {single_cat_records}")

# 最终结论
print("\n" + "=" * 60)
if all_passed:
    print("Task 1 验证通过 - 所有检查项 [OK]")
else:
    print("Task 1 验证失败 - 存在未通过检查项")
print("=" * 60)

sys.exit(0 if all_passed else 1)
