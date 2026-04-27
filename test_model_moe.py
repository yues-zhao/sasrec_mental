# test_model_moe.py
"""测试Task 4: SASRecMoE模型"""
import sys
import torch
import numpy as np
import argparse

print("=" * 60)
print("Task 4 测试SASRecMoE模型")
print("=" * 60)

all_passed = True

# 配置参数
batch_size = 4
maxlen = 10
hidden_units = 32
num_heads = 2
num_blocks = 2
dropout_rate = 0.5
usernum = 100
itemnum = 50
cat_num = 20

# 创建模拟args
class Args:
    def __init__(self):
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.num_accounts = 8
        self.top_n = 2
        self.temperature = 1.0
        self.cat_emb_dim = 16
        self.beta = 0.5
        self.lambda_reg = 0.01

args = Args()

# 1. 测试SASRecMoE初始化
print("\n1. 测试SASRecMoE初始化...")
try:
    from model import SASRecMoE
    
    model = SASRecMoE(usernum, itemnum, cat_num, args)
    
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   总参数数: {param_count}")
    print(f"   可训练参数数: {trainable_count}")
    
    # 验证beta参数
    if isinstance(model.beta, nn.Parameter):
        print(f"   beta值: {model.beta.item():.2f} [OK]")
    else:
        print(f"   beta参数 [FAIL]")
        all_passed = False
    
    # 验证lambda_reg
    if model.lambda_reg > 0:
        print(f"   lambda_reg: {model.lambda_reg} [OK]")
    else:
        print(f"   lambda_reg [FAIL]")
        all_passed = False
    
    # 验证价格塔
    if hasattr(model, 'price_tower'):
        print(f"   价格塔 [OK]")
    else:
        print(f"   价格塔 [FAIL]")
        all_passed = False

except Exception as e:
    print(f"   测试失败 [FAIL]: {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

# 2. 测试品类偏离度在线计算（防止数据泄露）
print("\n2. 测试品类偏离度在线计算...")
try:
    cat_ids = torch.randint(1, cat_num + 1, (batch_size, maxlen))
    seq_cat_ids = cat_ids.clone()
    
    cat_dev = model.compute_cat_deviation(cat_ids, seq_cat_ids)
    
    if cat_dev.shape == (batch_size, maxlen):
        print(f"   品类偏离度形状: {cat_dev.shape} [OK]")
    else:
        print(f"   品类偏离度形状: {cat_dev.shape} [FAIL]")
        all_passed = False
    
    # 验证第一个位置的偏离度为0（防止数据泄露）
    first_pos_dev = cat_dev[:, 0]
    if torch.allclose(first_pos_dev, torch.zeros_like(first_pos_dev), atol=1e-6):
        print(f"   数据泄露防护: 第一个位置偏离度为0 [OK]")
    else:
        print(f"   数据泄露防护 [FAIL]")
        all_passed = False

except Exception as e:
    print(f"   测试失败 [FAIL]: {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

# 3. 测试forward方法（带特征）
print("\n3. 测试forward方法（带特征）...")
try:
    input_seq = torch.randint(0, itemnum + 1, (batch_size, maxlen))
    pos = torch.randint(0, itemnum + 1, (batch_size, maxlen))
    neg = torch.randint(0, itemnum + 1, (batch_size, maxlen))
    
    price_seq = torch.randn(batch_size, maxlen)
    price_pos = torch.randn(batch_size, maxlen)
    price_neg = torch.randn(batch_size, maxlen)
    
    cat_seq = torch.randint(0, cat_num + 1, (batch_size, maxlen))
    cat_pos = torch.randint(0, cat_num + 1, (batch_size, maxlen))
    cat_neg = torch.randint(0, cat_num + 1, (batch_size, maxlen))
    
    price_dev_seq = torch.randn(batch_size, maxlen)
    price_dev_pos = torch.randn(batch_size, maxlen)
    price_dev_neg = torch.randn(batch_size, maxlen)
    
    time_dev_seq = torch.randn(batch_size, maxlen)
    time_dev_pos = torch.randn(batch_size, maxlen)
    time_dev_neg = torch.randn(batch_size, maxlen)
    
    loss, auc = model(
        input_seq, pos, neg,
        price_seq, price_pos, price_neg,
        cat_seq, cat_pos, cat_neg,
        price_dev_seq, price_dev_pos, price_dev_neg,
        time_dev_seq, time_dev_pos, time_dev_neg,
        is_training=True
    )
    
    if loss.dim() == 0 and loss.item() > 0:
        print(f"   Loss: {loss.item():.4f} [OK]")
    else:
        print(f"   Loss [FAIL]: {loss}")
        all_passed = False
    
    if 0 <= auc.item() <= 1:
        print(f"   AUC: {auc.item():.4f} [OK]")
    else:
        print(f"   AUC [FAIL]: {auc.item()}")
        all_passed = False

except Exception as e:
    print(f"   测试失败 [FAIL]: {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

# 4. 测试梯度流
print("\n4. 测试梯度流...")
try:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    
    loss, auc = model(
        input_seq, pos, neg,
        price_seq, price_pos, price_neg,
        cat_seq, cat_pos, cat_neg,
        price_dev_seq, price_dev_pos, price_dev_neg,
        time_dev_seq, time_dev_pos, time_dev_neg,
        is_training=True
    )
    
    loss.backward()
    
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
    if has_grad:
        print("   梯度流 [OK]")
    else:
        print("   梯度流 [FAIL]")
        all_passed = False
    
    optimizer.step()
    print("   优化器更新 [OK]")

except Exception as e:
    print(f"   测试失败 [FAIL]: {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

# 5. 测试predict方法
print("\n5. 测试predict方法...")
try:
    input_seq = torch.randint(0, itemnum + 1, (1, maxlen))
    item_idx = torch.randint(1, itemnum + 1, (101,))
    
    logits = model.predict(input_seq, item_idx)
    
    if logits.shape == (101,):
        print(f"   预测得分形状: {logits.shape} [OK]")
    else:
        print(f"   预测得分形状: {logits.shape} [FAIL]")
        all_passed = False

except Exception as e:
    print(f"   测试失败 [FAIL]: {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

# 6. 测试双塔融合逻辑
print("\n6. 测试双塔融合逻辑...")
try:
    # 验证beta对得分的影响
    beta_value = model.beta.item()
    
    # 临时修改beta
    original_beta = model.beta.clone()
    model.beta.data = torch.tensor(0.0)
    
    loss_zero_beta, _ = model(
        input_seq.squeeze(0).unsqueeze(0).expand(batch_size, -1),
        pos, neg,
        price_seq, price_pos, price_neg,
        cat_seq, cat_pos, cat_neg,
        price_dev_seq, price_dev_pos, price_dev_neg,
        time_dev_seq, time_dev_pos, time_dev_neg,
        is_training=True
    )
    
    model.beta.data = torch.tensor(1.0)
    
    loss_one_beta, _ = model(
        input_seq.squeeze(0).unsqueeze(0).expand(batch_size, -1),
        pos, neg,
        price_seq, price_pos, price_neg,
        cat_seq, cat_pos, cat_neg,
        price_dev_seq, price_dev_pos, price_dev_neg,
        time_dev_seq, time_dev_pos, time_dev_neg,
        is_training=True
    )
    
    model.beta.data = original_beta
    
    if loss_zero_beta.item() != loss_one_beta.item():
        print(f"   beta=0: {loss_zero_beta.item():.4f}, beta=1: {loss_one_beta.item():.4f} [OK]")
    else:
        print(f"   双塔融合 [FAIL]")
        all_passed = False

except Exception as e:
    print(f"   测试失败 [FAIL]: {e}")
    all_passed = False

# 最终结论
import torch.nn as nn
print("\n" + "=" * 60)
if all_passed:
    print("Task 4 验证通过 - 所有检查项 [OK]")
else:
    print("Task 4 验证失败 - 存在未通过检查项")
print("=" * 60)

sys.exit(0 if all_passed else 1)
