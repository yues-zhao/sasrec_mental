# test_moe_modules.py
"""测试Task 3: MoE模块功能"""
import sys
import torch
import numpy as np

print("=" * 60)
print("Task 3 测试MoE模块功能")
print("=" * 60)

all_passed = True

# 配置参数
batch_size = 4
maxlen = 10
hidden_units = 32
cat_num = 100
cat_emb_dim = 16
num_accounts = 8
top_n = 2
temperature = 0.5

# 1. 测试情景向量构建模块
print("\n1. 测试情景向量构建模块...")
try:
    from modules.moe_modules import ScenarioVectorBuilder
    
    builder = ScenarioVectorBuilder(hidden_units, cat_num, cat_emb_dim)
    
    price = torch.randn(batch_size, maxlen)
    price_dev = torch.randn(batch_size, maxlen)
    time_dev = torch.randn(batch_size, maxlen)
    cat_dev = torch.randn(batch_size, maxlen)
    cat_ids = torch.randint(1, cat_num + 1, (batch_size, maxlen))
    
    scenario_vectors = builder(price, price_dev, time_dev, cat_dev, cat_ids)
    
    if scenario_vectors.shape == (batch_size, maxlen, hidden_units):
        print(f"   输出形状: {scenario_vectors.shape} [OK]")
    else:
        print(f"   输出形状: {scenario_vectors.shape} [FAIL]")
        all_passed = False

except Exception as e:
    print(f"   测试失败 [FAIL]: {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

# 2. 测试可学习账户原型与门控机制
print("\n2. 测试可学习账户原型与门控机制...")
try:
    from modules.moe_modules import AccountPrototype
    
    prototype = AccountPrototype(num_accounts, hidden_units)
    
    scenario_flat = torch.randn(batch_size * maxlen, hidden_units)
    
    alpha, top_indices = prototype(scenario_flat, temperature, top_n)
    
    if alpha.shape == (batch_size * maxlen, num_accounts):
        print(f"   Alpha形状: {alpha.shape} [OK]")
    else:
        print(f"   Alpha形状: {alpha.shape} [FAIL]")
        all_passed = False
    
    if top_indices.shape == (batch_size * maxlen, top_n):
        print(f"   Top索引形状: {top_indices.shape} [OK]")
    else:
        print(f"   Top索引形状: {top_indices.shape} [FAIL]")
        all_passed = False
    
    non_zero_count = (alpha > 1e-6).sum(dim=-1)
    if torch.all(non_zero_count == top_n):
        print(f"   稀疏性验证: 每行{top_n}个非零值 [OK]")
    else:
        print(f"   稀疏性验证 [FAIL]")
        all_passed = False
    
    row_sums = alpha.sum(dim=-1)
    if torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
        print(f"   归一化验证: 每行和为1 [OK]")
    else:
        print(f"   归一化验证 [FAIL]")
        all_passed = False

except Exception as e:
    print(f"   测试失败 [FAIL]: {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

# 3. 测试MoE专家网络
print("\n3. 测试MoE专家网络...")
try:
    from modules.moe_modules import ExpertNetwork
    
    expert = ExpertNetwork(hidden_units, cat_emb_dim)
    
    expert_price = torch.randn(batch_size * maxlen, 1)
    expert_cat_emb = torch.randn(batch_size * maxlen, cat_emb_dim)
    
    preference = expert(expert_price, expert_cat_emb)
    
    if preference.shape == (batch_size * maxlen, 1):
        print(f"   输出形状: {preference.shape} [OK]")
    else:
        print(f"   输出形状: {preference.shape} [FAIL]")
        all_passed = False

except Exception as e:
    print(f"   测试失败 [FAIL]: {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

# 4. 测试完整的MoE价格塔
print("\n4. 测试完整的MoE价格塔...")
try:
    from modules.moe_modules import MoEPriceTower
    
    moe = MoEPriceTower(hidden_units, cat_num, num_accounts=num_accounts, top_n=top_n, temperature=temperature, cat_emb_dim=cat_emb_dim)
    
    price = torch.randn(batch_size, maxlen)
    price_dev = torch.randn(batch_size, maxlen)
    time_dev = torch.randn(batch_size, maxlen)
    cat_dev = torch.randn(batch_size, maxlen)
    cat_ids = torch.randint(1, cat_num + 1, (batch_size, maxlen))
    
    s_price, alpha, top_indices = moe(price, price_dev, time_dev, cat_dev, cat_ids)
    
    if s_price.shape == (batch_size * maxlen,):
        print(f"   s_price形状: {s_price.shape} [OK]")
    else:
        print(f"   s_price形状: {s_price.shape} [FAIL]")
        all_passed = False
    
    reg_loss = moe.compute_reg_loss(alpha)
    
    if reg_loss.dim() == 0 and reg_loss.item() > 0:
        print(f"   防坍缩正则项: {reg_loss.item():.4f} [OK]")
    else:
        print(f"   防坍缩正则项 [FAIL]")
        all_passed = False
    
    loss = s_price.mean() + reg_loss
    loss.backward()
    
    has_grad = any(p.grad is not None for p in moe.parameters())
    if has_grad:
        print("   梯度流验证 [OK]")
    else:
        print("   梯度流验证 [FAIL]")
        all_passed = False

except Exception as e:
    print(f"   测试失败 [FAIL]: {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

# 5. 测试各专家独立性
print("\n5. 测试各专家独立性...")
try:
    expert_params = []
    for expert in moe.experts:
        expert_params.append(id(expert.network[0].weight))
    
    unique_params = len(set(expert_params))
    if unique_params == moe.num_accounts:
        print(f"   专家独立性: {unique_params}个独立参数 [OK]")
    else:
        print(f"   专家独立性 [FAIL]")
        all_passed = False

except Exception as e:
    print(f"   测试失败 [FAIL]: {e}")
    all_passed = False

# 最终结论
print("\n" + "=" * 60)
if all_passed:
    print("Task 3 验证通过 - 所有检查项 [OK]")
else:
    print("Task 3 验证失败 - 存在未通过检查项")
print("=" * 60)

sys.exit(0 if all_passed else 1)
