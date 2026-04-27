# enhanced_preprocess.py
import pandas as pd
import numpy as np
from collections import defaultdict
import json
import os

# ===================== 1. 配置路径 =====================
RAW_DATA_PATH = "data/ta_feng_all_months_merged.csv"
OUTPUT_PATH = "data/TaFeng_MoE.txt"
CAT_ITEM_MAP_PATH = "data/cat_item_map.json"
FEATURE_STATS_PATH = "data/feature_stats.json"

# 过滤阈值
MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 5


# ===================== 2. 读取并清洗数据 =====================
print("正在读取原始数据...")
try:
    df = pd.read_csv(RAW_DATA_PATH, encoding="utf-8")
except UnicodeDecodeError:
    try:
        df = pd.read_csv(RAW_DATA_PATH, encoding="latin1")
    except:
        df = pd.read_csv(RAW_DATA_PATH, encoding="gbk")

# 只保留核心列：用户ID、商品ID、日期、价格、品类
df = df[["CUSTOMER_ID", "PRODUCT_ID", "TRANSACTION_DT", "SALES_PRICE", "PRODUCT_SUBCLASS"]].copy()
df.columns = ["user_id", "item_id", "date", "price", "category"]

# 去重：同一用户同一天买同一件商品，只留1条记录
df = df.drop_duplicates(subset=["user_id", "item_id", "date"])

# 丢弃空值
df = df.dropna()

# 转换日期为可排序格式（原始格式：MM/DD/YYYY）
def parse_date(date_str):
    try:
        parts = str(date_str).split('/')
        if len(parts) == 3:
            month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
            return year * 10000 + month * 100 + day
    except:
        return 0
    return 0

df['date_numeric'] = df['date'].apply(parse_date)

print(f"原始数据量: {len(df)} 条记录")


# ===================== 3. 统计交互次数（过滤低频） =====================
print("正在过滤低频用户和商品...")
countU = defaultdict(int)
countP = defaultdict(int)

for _, row in df.iterrows():
    user = row["user_id"]
    item = row["item_id"]
    countU[user] += 1
    countP[item] += 1


# ===================== 4. 构建用户序列（按时间排序） =====================
print("正在构建用户序列...")
usermap = {}
itemmap = {}
catmap = {}
usernum = 0
itemnum = 0
catnum = 0

# 按用户分组，按时间排序
user_seqs = defaultdict(list)

for _, row in df.iterrows():
    raw_user = row["user_id"]
    raw_item = row["item_id"]
    date = row["date_numeric"]
    price = row["price"]
    cat = row["category"]
    
    # 过滤低频
    if countU[raw_user] < MIN_USER_INTERACTIONS or countP[raw_item] < MIN_ITEM_INTERACTIONS:
        continue
    
    # 用户ID映射
    if raw_user not in usermap:
        usernum += 1
        usermap[raw_user] = usernum
    uid = usermap[raw_user]
    
    # 商品ID映射
    if raw_item not in itemmap:
        itemnum += 1
        itemmap[raw_item] = itemnum
    iid = itemmap[raw_item]
    
    # 品类ID映射
    if cat not in catmap:
        catnum += 1
        catmap[cat] = catnum
    cid = catmap[cat]
    
    # 保存：(日期, 商品ID, 价格, 品类ID)
    user_seqs[uid].append((date, iid, price, cid))

# 按时间排序
for uid in user_seqs:
    user_seqs[uid].sort(key=lambda x: x[0])

print(f"有效用户数: {usernum}")
print(f"有效商品数: {itemnum}")
print(f"有效品类数: {catnum}")


# ===================== 5. 品类内价格对数压缩 =====================
print("正在进行品类内价格对数压缩...")

# 构建品类到商品列表的映射（用于负采样）
cat_to_items = defaultdict(set)
# 构建品类内所有价格列表（用于归一化）
cat_prices = defaultdict(list)

for uid in user_seqs:
    for date, iid, price, cid in user_seqs[uid]:
        cat_to_items[cid].add(iid)
        cat_prices[cid].append(price)

# 转换为numpy数组并排序（用于计算分位数归一化）
cat_price_stats = {}
for cid in cat_prices:
    prices = np.array(cat_prices[cid], dtype=np.float64)
    # 对数压缩
    log_prices = np.log1p(prices)
    cat_price_stats[cid] = {
        'mean': float(np.mean(log_prices)),
        'std': float(np.std(log_prices)),
        'min': float(np.min(log_prices)),
        'max': float(np.max(log_prices))
    }

print(f"品类价格统计信息计算完成")


# ===================== 6. 离线计算价格偏离度（防止数据泄露） =====================
print("正在计算价格偏离度（离线，防止数据泄露）...")

def compute_price_deviation(seq, item_idx):
    """
    计算第item_idx个物品的价格偏离度
    严格防止数据泄露：只使用item_idx之前的历史数据
    
    参数:
        seq: [(date, iid, price, cid), ...] 按时间排序的用户序列
        item_idx: 当前物品在序列中的索引
    
    返回:
        price_deviation: 价格偏离度（Z-score归一化）
        normalized_price: 品类内归一化后的价格（对数压缩后）
    """
    if item_idx <= 0:
        return 0.0, 0.0
    
    current_date, current_iid, current_price, current_cid = seq[item_idx]
    
    # 收集历史同品类价格（只使用当前时刻之前的数据）
    historical_prices = []
    for i in range(item_idx):
        hist_date, hist_iid, hist_price, hist_cid = seq[i]
        if hist_cid == current_cid:
            historical_prices.append(hist_price)
    
    if len(historical_prices) < 2:
        # 历史数据不足，使用当前品类的全局统计信息
        stats = cat_price_stats.get(current_cid, {'mean': 0, 'std': 1})
        current_log_price = np.log1p(current_price)
        normalized_price = (current_log_price - stats['mean']) / (stats['std'] + 1e-8)
        return 0.0, float(normalized_price)
    
    # 计算历史同品类价格统计信息
    historical_prices = np.array(historical_prices, dtype=np.float64)
    log_hist_prices = np.log1p(historical_prices)
    hist_mean = np.mean(log_hist_prices)
    hist_std = np.std(log_hist_prices)
    
    # 当前物品价格对数压缩
    current_log_price = np.log1p(current_price)
    
    # Z-score归一化
    if hist_std < 1e-8:
        price_deviation = 0.0
    else:
        price_deviation = (current_log_price - hist_mean) / hist_std
    
    # 使用全局品类统计信息进行价格归一化
    stats = cat_price_stats.get(current_cid, {'mean': 0, 'std': 1})
    normalized_price = (current_log_price - stats['mean']) / (stats['std'] + 1e-8)
    
    return float(price_deviation), float(normalized_price)


# ===================== 7. 离线计算时间偏离度（防止数据泄露） =====================
print("正在计算时间偏离度（离线，防止数据泄露）...")

def compute_time_deviation(seq, item_idx):
    """
    计算第item_idx个物品的时间偏离度
    严格防止数据泄露：只使用item_idx之前的历史数据
    
    参数:
        seq: [(date, iid, price, cid), ...] 按时间排序的用户序列
        item_idx: 当前物品在序列中的索引
    
    返回:
        time_deviation: 时间偏离度（Z-score归一化）
    """
    if item_idx <= 0:
        return 0.0
    
    current_date, _, _, _ = seq[item_idx]
    
    # 计算历史时间间隔序列（只使用当前时刻之前的数据）
    historical_intervals = []
    for i in range(1, item_idx + 1):
        prev_date, _, _, _ = seq[i - 1]
        curr_date, _, _, _ = seq[i]
        interval = abs(curr_date - prev_date)
        historical_intervals.append(interval)
    
    if len(historical_intervals) < 2:
        return 0.0
    
    # 当前时间间隔
    current_interval = historical_intervals[-1]
    
    # 使用之前的历史间隔计算统计信息（防止数据泄露）
    prev_intervals = np.array(historical_intervals[:-1], dtype=np.float64)
    hist_mean = np.mean(prev_intervals)
    hist_std = np.std(prev_intervals)
    
    if hist_std < 1e-8:
        time_deviation = 0.0
    else:
        time_deviation = (current_interval - hist_mean) / hist_std
    
    return float(time_deviation)


# ===================== 8. 构建增强序列并输出 =====================
print("正在构建增强序列并输出...")

# 输出格式：用户ID 商品ID 价格(归一化) 品类ID 价格偏离度 时间偏离度
# 同时记录日期用于验证

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for uid in user_seqs:
        seq = user_seqs[uid]
        
        for item_idx in range(len(seq)):
            date, iid, price, cid = seq[item_idx]
            
            # 计算价格偏离度和归一化价格
            price_dev, norm_price = compute_price_deviation(seq, item_idx)
            
            # 计算时间偏离度
            time_dev = compute_time_deviation(seq, item_idx)
            
            # 写入：用户ID 商品ID 归一化价格 品类ID 价格偏离度 时间偏离度
            f.write(f"{uid} {iid} {norm_price:.6f} {cid} {price_dev:.6f} {time_dev:.6f}\n")


# ===================== 9. 保存品类到商品映射（用于负采样） =====================
print("正在保存品类到商品映射...")

cat_item_map_dict = {}
all_item_list = sorted(set(iid for uid in user_seqs for _, iid, _, _ in user_seqs[uid]))

for cid, items in cat_to_items.items():
    cat_item_map_dict[str(cid)] = sorted(list(items))

# 保存映射和全局商品列表
sampling_data = {
    'cat_item_map': cat_item_map_dict,
    'all_item_list': all_item_list,
    'single_item_cats': [cid for cid, items in cat_item_map_dict.items() if len(items) == 1]
}

with open(CAT_ITEM_MAP_PATH, "w", encoding="utf-8") as f:
    json.dump(sampling_data, f)


# ===================== 10. 保存特征统计信息 =====================
print("正在保存特征统计信息...")

feature_stats = {
    'user_num': usernum,
    'item_num': itemnum,
    'cat_num': catnum,
    'cat_price_stats': cat_price_stats,
    'cat_item_map': {k: len(v) for k, v in cat_item_map_dict.items()}
}

with open(FEATURE_STATS_PATH, "w", encoding="utf-8") as f:
    json.dump(feature_stats, f)


# ===================== 11. 输出统计信息 =====================
print(f"\n{'='*60}")
print(f"增强预处理完成！")
print(f"{'='*60}")
print(f"有效用户数: {usernum}")
print(f"有效商品数: {itemnum}")
print(f"有效品类数: {catnum}")
print(f"输出文件: {OUTPUT_PATH}")
print(f"品类-商品映射: {CAT_ITEM_MAP_PATH}")
print(f"特征统计信息: {FEATURE_STATS_PATH}")
print(f"{'='*60}")

# 验证输出
with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()
    print(f"\n输出文件行数: {len(lines)}")
    print(f"前5行示例:")
    for line in lines[:5]:
        parts = line.strip().split()
        if len(parts) == 6:
            uid, iid, price, cid, price_dev, time_dev = parts
            print(f"  用户={uid}, 商品={iid}, 价格={price}, 品类={cid}, 价格偏离={price_dev}, 时间偏离={time_dev}")
