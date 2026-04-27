# test_task5.py
"""测试Task 5: 训练流程支持新模型"""
import sys
import subprocess
import os

print("=" * 60)
print("Task 5 测试训练流程")
print("=" * 60)

all_passed = True

# 1. 测试向后兼容性（原始SASRec模型加载）
print("\n1. 测试向后兼容性（原始SASRec模型加载）...")
try:
    from model import SASRec
    
    class MockArgs:
        def __init__(self):
            self.hidden_units = 50
            self.maxlen = 50
            self.num_blocks = 2
            self.num_heads = 1
            self.dropout_rate = 0.5
    
    args = MockArgs()
    sasrec = SASRec(100, 100, args)
    
    print(f"   原始SASRec模型加载 [OK]")
except Exception as e:
    print(f"   测试失败 [FAIL]: {e}")
    all_passed = False

# 2. 测试MoE参数解析
print("\n2. 测试MoE参数解析...")
try:
    cmd = 'cd d:\\trae_projects\\SAS_pytorch && python -c "import main"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    
    # Just check if imports work
    from main import str2bool
    
    if str2bool('True') == True and str2bool('False') == False:
        print(f"   str2bool函数 [OK]")
    else:
        print(f"   str2bool函数 [FAIL]")
        all_passed = False
    
    print(f"   main.py模块导入 [OK]")

except Exception as e:
    print(f"   测试失败 [FAIL]: {e}")
    all_passed = False

# 3. 测试参数配置完整性
print("\n3. 测试参数配置完整性...")
try:
    # Verify that main.py has MoE arguments
    with open('main.py', 'r', encoding='utf-8') as f:
        main_content = f.read()
    
    required_moe_args = ['use_moe', 'cat_num', 'num_accounts', 'top_n', 'temperature', 'beta', 'lambda_reg']
    missing_args = [arg for arg in required_moe_args if arg not in main_content]
    
    if len(missing_args) == 0:
        print(f"   MoE参数完整性 [OK] - {len(required_moe_args)}个参数均已定义")
    else:
        print(f"   MoE参数完整性 [FAIL] - 缺少: {missing_args}")
        all_passed = False

except Exception as e:
    print(f"   测试失败 [FAIL]: {e}")
    all_passed = False

# 4. 测试数据加载模块集成
print("\n4. 测试数据加载模块集成...")
try:
    from util import data_partition
    from sampler import WarpSampler
    
    # 测试带特征的数据加载
    dataset = data_partition("TaFeng_MoE", use_features=True)
    if len(dataset) == 8:
        user_train, user_valid, user_test, usernum, itemnum, \
        user_train_features, user_valid_features, user_test_features = dataset
        
        print(f"   多特征数据加载 [OK]")
        print(f"   用户数: {usernum}, 商品数: {itemnum}")
    else:
        print(f"   多特征数据加载 [FAIL]")
        all_passed = False

except Exception as e:
    print(f"   测试失败 [FAIL]: {e}")
    all_passed = False

# 5. 测试模型创建
print("\n5. 测试模型创建...")
try:
    from model import SASRec, SASRecMoE
    
    class MockArgs:
        def __init__(self):
            self.hidden_units = 32
            self.maxlen = 50
            self.num_blocks = 2
            self.num_heads = 1
            self.dropout_rate = 0.5
            self.num_accounts = 8
            self.top_n = 2
            self.temperature = 1.0
            self.cat_emb_dim = 16
            self.beta = 0.5
            self.lambda_reg = 0.01
    
    args = MockArgs()
    
    # 测试SASRec
    sasrec = SASRec(100, 50, args)
    sasrec_params = sum(p.numel() for p in sasrec.parameters())
    print(f"   SASRec参数数: {sasrec_params} [OK]")
    
    # 测试SASRecMoE
    sasrec_moe = SASRecMoE(100, 50, 20, args)
    sasrec_moe_params = sum(p.numel() for p in sasrec_moe.parameters())
    print(f"   SASRecMoE参数数: {sasrec_moe_params} [OK]")

except Exception as e:
    print(f"   测试失败 [FAIL]: {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

# 最终结论
print("\n" + "=" * 60)
if all_passed:
    print("Task 5 验证通过 - 所有检查项 [OK]")
else:
    print("Task 5 验证失败 - 存在未通过检查项")
print("=" * 60)

sys.exit(0 if all_passed else 1)
