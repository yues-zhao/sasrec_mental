# tafeng_preprocess.py
import pandas as pd
from collections import defaultdict

# ===================== 1. 配置路径 =====================
# 1. 把下载的 CSV 文件放在项目根目录
RAW_DATA_PATH = "data/ta_feng_all_months_merged.csv"
# 2. 输出路径（必须放在 data 文件夹下，和原项目一致）
OUTPUT_PATH = "data/TaFeng.txt"

# ===================== 2. 读取并清洗数据 =====================
# 读取CSV（尝试utf-8编码）
try:
    df = pd.read_csv(RAW_DATA_PATH, encoding="utf-8")
except UnicodeDecodeError:
    try:
        df = pd.read_csv(RAW_DATA_PATH, encoding="latin1")
    except:
        df = pd.read_csv(RAW_DATA_PATH, encoding="gbk")
# 只保留核心3列：用户ID、商品ID、日期
df = df[["CUSTOMER_ID", "PRODUCT_ID", "TRANSACTION_DT"]].copy()
# 去重：同一用户同一天买同一件商品，只留1条记录
df = df.drop_duplicates(subset=["CUSTOMER_ID", "PRODUCT_ID", "TRANSACTION_DT"])
# 丢弃空值
df = df.dropna()

# ===================== 3. 统计交互次数（过滤低频） =====================
countU = defaultdict(int)  # 用户交互次数
countP = defaultdict(int)  # 商品被交互次数
for _, row in df.iterrows():
    user = str(row["CUSTOMER_ID"])
    item = str(row["PRODUCT_ID"])
    countU[user] += 1
    countP[item] += 1

# ===================== 4. 原始ID → 连续整型ID（模型必须要！） =====================
usermap = {}  # 原始用户ID → 新ID(1,2,3...)
itemmap = {}  # 原始商品ID → 新ID(1,2,3...)
usernum = 0
itemnum = 0

# 按用户分组 + 按日期排序（序列推荐核心：时间顺序）
user_seqs = defaultdict(list)
for _, row in df.iterrows():
    raw_user = str(row["CUSTOMER_ID"])
    raw_item = str(row["PRODUCT_ID"])
    date = row["TRANSACTION_DT"]

    # 过滤：交互<5次的用户/商品直接丢弃
    if countU[raw_user] < 5 or countP[raw_item] < 5:
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

    # 保存：(日期, 商品ID)，用于排序
    user_seqs[uid].append((date, iid))

# ===================== 5. 按时间排序 + 输出最终文件 =====================
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for uid in user_seqs:
        # 按日期升序排序（用户的购物历史顺序）
        user_seqs[uid].sort(key=lambda x: x[0])
        # 只保留商品ID，写入文件
        for date, iid in user_seqs[uid]:
            f.write(f"{uid} {iid}\n")

print(f"预处理完成！")
print(f"有效用户数：{usernum}, 有效商品数：{itemnum}")
print(f"输出文件：{OUTPUT_PATH}")