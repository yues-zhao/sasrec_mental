# SASRec价格感知MoE增强模块规格

## Why
在现有SASRec模型基础上引入价格感知能力，通过Mixture of Experts (MoE)架构建模用户的价格偏好。原始SASRec仅学习用户兴趣表示，新增的价格塔将捕捉用户对不同价格账户的偏好，两者结合提升推荐质量。

## What Changes
- **新增** `data/enhanced_preprocess.py`：增强版数据预处理脚本，提取价格、时间、品类信息
- **新增** `data/features.py`：离线特征计算模块（价格偏离度、时间偏离度）
- **新增** `modules/moe_modules.py`：MoE价格塔模块（门控、专家网络、账户原型）
- **修改** `model.py`：从SASRec改为SASRecMoE，集成兴趣塔和价格塔
- **修改** `util.py`：更新数据加载和评估逻辑，支持新特征
- **修改** `sampler.py`：扩展采样器，支持多模态特征输入
- **修改** `main.py`：支持新模型参数和训练流程

## Impact
- 影响模型：SASRec → SASRecMoE
- 影响数据流：需要原始CSV中的价格、时间、品类信息
- 影响训练：新增MoE相关超参数和损失计算
- 向后兼容：**BREAKING** - 新模型不兼容旧模型检查点

## ADDED Requirements

### Requirement 1: 增强数据预处理
系统必须从ta_feng_all_months_merged.csv中提取并计算以下信息：
- 原始交互序列（用户ID, 商品ID, 日期, 价格, 品类）
- 品类内对数压缩后的价格
- 离线特征：价格偏离度、时间偏离度
- 在线特征所需的品类信息

#### Scenario: 成功预处理
- **WHEN** 用户运行 `python data/enhanced_preprocess.py`
- **THEN** 生成包含所有必要特征的预处理文件
- **AND** 严格防止数据泄露（计算t时刻特征仅使用t时刻之前数据）
- **AND** 输出数据格式：`用户ID 商品ID 价格 品类 价格偏离度 时间偏离度 日期`

### Requirement 2: 价格偏离度特征
系统必须计算价格偏离度：
- 公式：`价格偏离度 = (当前价格 - 历史同品类平均价格) / 历史同品类价格标准差`
- 严格防止数据泄露：计算t时刻时仅使用t时刻之前的历史交互
- 对结果进行Z-score归一化

#### Scenario: 计算价格偏离度
- **WHEN** 处理用户序列中的第t个物品
- **THEN** 仅使用该用户前t-1个交互中相同品类的价格
- **AND** 计算历史同品类平均价格和标准差
- **AND** 输出归一化的价格偏离度

### Requirement 3: 时间偏离度特征
系统必须计算时间偏离度：
- 公式：`时间偏离度 = (当前时间间隔 - 历史平均时间间隔) / 历史时间间隔标准差`
- 严格防止数据泄露
- 对结果进行Z-score归一化

#### Scenario: 计算时间偏离度
- **WHEN** 处理用户序列中的第t个物品
- **THEN** 计算与前一个交互的时间间隔
- **AND** 使用历史时间间隔统计信息进行归一化

### Requirement 4: 品类偏离度特征（在线计算）
系统必须在线计算品类偏离度：
- 公式：`品类偏离度 = 1 - cosine(当前品类嵌入, 历史序列品类嵌入的平均池化)`
- 严格防止数据泄露
- 需要品类嵌入表

#### Scenario: 计算品类偏离度
- **WHEN** 模型前向传播时
- **THEN** 获取当前物品的品类嵌入
- **AND** 获取历史序列中所有物品的品类嵌入并计算平均池化
- **AND** 计算cosine相似度，输出品类偏离度

### Requirement 5: 情景向量构建
系统必须构建情景向量：
- 输入：当前物品价格（品类内归一化后）、价格偏离度、时间偏离度、品类偏离度（4个标量）
- 经过线性网络映射为高维向量
- 与当前物品的品类嵌入拼接
- 输出：情景向量

#### Scenario: 构建情景向量
- **WHEN** 前向传播处理当前物品
- **THEN** 将4个标量拼接为向量 [price, price_dev, time_dev, cat_dev]
- **AND** 通过Linear(4, hidden_dim)映射
- **AND** 与品类嵌入拼接
- **THEN** 输出情景向量

### Requirement 6: 可学习账户原型与门控机制
系统必须实现：
- 初始化K个可学习的账户原型向量
- 计算情景向量与每个账户的相似度
- 使用温度系数控制softmax分布
- 输出α（分配到每个账户的概率）
- 支持TopN稀疏门控（N通常取1或2）

#### Scenario: 计算门控权重
- **WHEN** 输入情景向量
- **THEN** 计算与K个账户原型的相似度
- **AND** 应用温度系数和softmax得到α
- **AND** 取TopN概率最大的索引
- **AND** 对TopN子集重归一化
- **THEN** 输出稀疏门控权重

### Requirement 7: MoE专家网络
系统必须构建MoE：
- K个同构但参数独立的专家网络
- 输入：当前物品价格（品类内归一化）+ 品类嵌入
- 每个专家输出基础偏好标量
- 仅对被激活的N个专家执行前向计算

#### Scenario: 执行MoE前向传播
- **WHEN** 获得稀疏门控权重
- **THEN** 仅对被激活的N个专家执行前向计算
- **AND** 每个专家输出基础偏好标量
- **AND** 结合门控权重计算加权平均
- **THEN** 输出整体价格账户得分 s_price

### Requirement 8: 双塔融合
系统必须将兴趣塔和价格塔结合：
- 兴趣塔得分：s_interest（原始SASRec输出）
- 价格塔得分：s_price（MoE输出）
- 总得分：s = s_interest + β × s_price
- β为可学习参数或超参数

#### Scenario: 计算最终得分
- **WHEN** 获得s_interest和s_price
- **THEN** 计算 s = s_interest + β * s_price
- **THEN** 用于最终的排序和损失计算

### Requirement 9: 混合负采样策略
系统必须实现混合负采样策略：
- **品类商品数 > 1**：使用品类内负采样
- **品类商品数 = 1**：使用全局随机负采样（从所有商品中随机抽取，避开正样本自身）
- 预处理阶段构建字典：`{cat_id: [item_id_list]}` 和 `all_item_list`
- DataLoader生成负样本时，先检查正样本品类的商品数量，再决定采样策略
- 选择的负样本不能等于正样本item_id

#### Scenario: 品类内负采样（品类商品数 > 1）
- **WHEN** DataLoader需要为某个正样本生成负样本
- **AND** 该正样本的品类商品数大于1
- **THEN** 获取正样本的品类ID (target_cat)
- **AND** 从该品类的商品列表中随机抽取一个商品
- **AND** 确保抽取的商品不等于正样本商品ID
- **THEN** 返回该负样本

#### Scenario: 全局负采样（品类商品数 = 1）
- **WHEN** DataLoader需要为某个正样本生成负样本
- **AND** 该正样本的品类商品数等于1（无法进行品类内采样）
- **THEN** 从所有商品列表中随机抽取一个商品
- **AND** 确保抽取的商品不等于正样本商品ID
- **THEN** 返回该负样本

#### Scenario: 构建品类到商品映射
- **WHEN** 执行数据预处理时
- **THEN** 遍历所有商品，构建 `{cat_id: [item_id_1, item_id_2, ...]}` 字典
- **AND** 构建 `all_item_list` 所有商品列表
- **AND** 保存映射供DataLoader使用

### Requirement 10: BPR损失函数与防坍缩正则项
系统必须使用以下损失函数：
- **BPR损失**：标准的Bayesian Personalized Ranking损失
- **防坍缩正则项**：`L_reg = -λ × Σ α × log(α)`，其中α为门控权重
- **总损失**：`L_total = L_BPR + L_reg`
- λ为正则化系数，可通过超参数配置

#### Scenario: 计算总损失
- **WHEN** 模型前向传播获得预测结果
- **THEN** 计算BPR损失：基于正负样本得分差异
- **AND** 计算防坍缩正则项：对门控权重α计算熵正则化
- **AND** 计算总损失：L_total = L_BPR + L_reg
- **THEN** 使用总损失进行反向传播

#### Scenario: 防坍缩机制
- **WHEN** 门控权重α出现坍缩（某些专家概率趋近于0）
- **THEN** 正则项鼓励更均匀的门控分布
- **AND** 防止部分专家在训练中被完全忽略
- **THEN** 保证所有专家都能参与学习

## MODIFIED Requirements

### Requirement: 数据加载模块
**修改内容**：`util.py` 和 `sampler.py`
- 支持加载带有多特征的数据格式
- 采样器需要返回用户序列、正负样本序列及其对应的特征

### Requirement: 模型训练流程
**修改内容**：`main.py`
- 新增MoE相关参数：num_experts, top_n, temperature, beta
- 保持原有训练循环结构
- 评估逻辑需要适应新的评分方式

## REMOVED Requirements
**无** - 所有原有功能保留，仅进行扩展
