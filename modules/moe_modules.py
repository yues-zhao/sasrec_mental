# -*- coding: utf-8 -*-
"""
MoE价格塔模块
包含：
- 情景向量构建
- 可学习账户原型与门控机制
- MoE专家网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScenarioVectorBuilder(nn.Module):
    """
    情景向量构建模块
    输入：当前物品价格（品类内归一化）、价格偏离度、时间偏离度、品类偏离度（4个标量）
    输出：情景向量 = Linear([price, price_dev, time_dev, cat_dev]) + 品类嵌入
    """
    
    def __init__(self, hidden_units, cat_num, cat_emb_dim=32):
        super(ScenarioVectorBuilder, self).__init__()
        
        self.hidden_units = hidden_units
        self.cat_emb_dim = cat_emb_dim
        
        # 4个标量 -> 高维向量的线性映射
        self.feature_projection = nn.Linear(4, hidden_units)
        
        # 品类嵌入表
        self.cat_embedding = nn.Embedding(cat_num + 1, cat_emb_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.cat_embedding.weight)
        
        # 拼接后映射到统一维度
        # (hidden_units + cat_emb_dim) -> hidden_units
        self.fusion = nn.Linear(hidden_units + cat_emb_dim, hidden_units)
        
        # Layer Norm
        self.layer_norm = nn.LayerNorm(hidden_units)
    
    def forward(self, price, price_dev, time_dev, cat_dev, cat_ids):
        """
        参数:
            price: (batch_size, maxlen) - 品类内归一化价格
            price_dev: (batch_size, maxlen) - 价格偏离度
            time_dev: (batch_size, maxlen) - 时间偏离度
            cat_dev: (batch_size, maxlen) - 品类偏离度
            cat_ids: (batch_size, maxlen) - 品类ID
        返回:
            scenario_vectors: (batch_size, maxlen, hidden_units)
        """
        batch_size, maxlen = price.size()
        
        # 拼接4个标量
        feature_vector = torch.stack([price, price_dev, time_dev, cat_dev], dim=-1)  # (batch_size, maxlen, 4)
        
        # 映射到高维
        feature_proj = self.feature_projection(feature_vector)  # (batch_size, maxlen, hidden_units)
        
        # 获取品类嵌入
        cat_emb = self.cat_embedding(cat_ids)  # (batch_size, maxlen, cat_emb_dim)
        
        # 拼接并融合
        combined = torch.cat([feature_proj, cat_emb], dim=-1)  # (batch_size, maxlen, hidden_units + cat_emb_dim)
        scenario_vectors = self.fusion(combined)
        scenario_vectors = self.layer_norm(scenario_vectors)
        
        return scenario_vectors


class AccountPrototype(nn.Module):
    """
    可学习的账户原型
    初始化K个可学习的账户原型向量
    """
    
    def __init__(self, num_accounts, hidden_units):
        super(AccountPrototype, self).__init__()
        
        self.num_accounts = num_accounts
        self.hidden_units = hidden_units
        
        # K个可学习的账户原型向量
        self.account_prototypes = nn.Parameter(torch.randn(num_accounts, hidden_units))
        nn.init.xavier_uniform_(self.account_prototypes)
    
    def forward(self, scenario_vectors, temperature=1.0, top_n=2):
        """
        计算情景向量与每个账户的相似度，得到门控权重α
        参数:
            scenario_vectors: (batch_size * maxlen, hidden_units) 或 (batch_size, maxlen, hidden_units)
            temperature: 温度系数，控制softmax分布
            top_n: 稀疏门控，只保留TopN个概率最大的账户
        返回:
            alpha: (batch_size * maxlen, num_accounts) - 门控权重（稀疏化后）
            gate_indices: (batch_size * maxlen, top_n) - 激活的账户索引
        """
        if scenario_vectors.dim() == 3:
            # 展平 (batch_size, maxlen, hidden_units) -> (batch_size * maxlen, hidden_units)
            scenario_flat = scenario_vectors.view(-1, self.hidden_units)
        else:
            scenario_flat = scenario_vectors
        
        # 计算情景向量与每个账户原型的相似度（余弦相似度）
        # 归一化
        scenario_norm = F.normalize(scenario_flat, dim=-1)
        account_norm = F.normalize(self.account_prototypes, dim=-1)
        
        # 余弦相似度 (batch_size * maxlen, num_accounts)
        similarity = torch.matmul(scenario_norm, account_norm.t()) / temperature
        
        # Softmax得到概率
        alpha = F.softmax(similarity, dim=-1)
        
        # TopN稀疏门控
        top_values, top_indices = torch.topk(alpha, top_n, dim=-1)
        
        # 对TopN子集重归一化
        top_sum = top_values.sum(dim=-1, keepdim=True)
        top_sum = torch.clamp(top_sum, min=1e-8)
        sparse_alpha = top_values / top_sum
        
        # 构建完整的稀疏门控权重
        sparse_alpha_full = torch.zeros_like(alpha)
        sparse_alpha_full.scatter_(-1, top_indices, sparse_alpha)
        
        return sparse_alpha_full, top_indices
    
    def get_reg_loss(self):
        """
        计算防坍缩正则项
        L_reg = -λ × Σ α × log(α)
        注意：这个正则项的实际计算在模型forward中进行，这里预留接口
        """
        pass


class ExpertNetwork(nn.Module):
    """
    单个专家网络
    输入：当前物品价格（品类内归一化）+ 品类嵌入
    输出：基础偏好标量
    """
    
    def __init__(self, hidden_units, cat_emb_dim=32):
        super(ExpertNetwork, self).__init__()
        
        # 输入维度：价格(1) + 品类嵌入(cat_emb_dim)
        input_dim = 1 + cat_emb_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.LayerNorm(hidden_units),
            nn.Linear(hidden_units, hidden_units // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_units // 2),
            nn.Linear(hidden_units // 2, 1)  # 输出标量
        )
        
        # 初始化
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, price, cat_emb):
        """
        参数:
            price: (batch_size * maxlen, 1) - 归一化价格
            cat_emb: (batch_size * maxlen, cat_emb_dim) - 品类嵌入
        返回:
            preference: (batch_size * maxlen, 1) - 基础偏好标量
        """
        # 拼接价格和品类嵌入
        input_vector = torch.cat([price, cat_emb], dim=-1)
        
        # 前向计算
        preference = self.network(input_vector)
        
        return preference


class MoEPriceTower(nn.Module):
    """
    MoE价格塔
    结合门控机制和K个专家网络，输出整体价格账户得分
    """
    
    def __init__(self, hidden_units, cat_num, num_accounts=8, top_n=2, 
                 temperature=1.0, cat_emb_dim=32):
        super(MoEPriceTower, self).__init__()
        
        self.hidden_units = hidden_units
        self.num_accounts = num_accounts
        self.top_n = top_n
        self.temperature = temperature
        self.cat_emb_dim = cat_emb_dim
        
        # 情景向量构建
        self.scenario_builder = ScenarioVectorBuilder(
            hidden_units, cat_num, cat_emb_dim
        )
        
        # 账户原型（门控）
        self.account_prototype = AccountPrototype(num_accounts, hidden_units)
        
        # K个专家网络（同构但参数独立）
        self.experts = nn.ModuleList([
            ExpertNetwork(hidden_units, cat_emb_dim)
            for _ in range(num_accounts)
        ])
        
        # 保存品类嵌入表引用（用于输入构建）
        self.cat_embedding = self.scenario_builder.cat_embedding
    
    def forward(self, price, price_dev, time_dev, cat_dev, cat_ids, 
                pos_price=None, pos_cat_ids=None):
        """
        参数:
            price: (batch_size, maxlen) - 品类内归一化价格
            price_dev: (batch_size, maxlen) - 价格偏离度
            time_dev: (batch_size, maxlen) - 时间偏离度
            cat_dev: (batch_size, maxlen) - 品类偏离度
            cat_ids: (batch_size, maxlen) - 品类ID
            pos_price: (batch_size * maxlen, 1) - 正样本价格（可选，用于专家输入）
            pos_cat_ids: (batch_size * maxlen,) - 正样本品类ID（可选，用于专家输入）
        返回:
            s_price: (batch_size * maxlen,) - 整体价格账户得分
            alpha: (batch_size * maxlen, num_accounts) - 门控权重
            top_indices: (batch_size * maxlen, top_n) - 激活的账户索引
        """
        batch_size, maxlen = price.size()
        
        # 1. 构建情景向量
        scenario_vectors = self.scenario_builder(price, price_dev, time_dev, cat_dev, cat_ids)
        
        # 展平
        scenario_flat = scenario_vectors.view(-1, self.hidden_units)
        
        # 2. 计算门控权重
        alpha, top_indices = self.account_prototype(
            scenario_flat, self.temperature, self.top_n
        )
        
        # 3. 获取品类嵌入用于专家输入
        if pos_cat_ids is not None:
            cat_emb = self.cat_embedding(pos_cat_ids)  # (batch_size * maxlen, cat_emb_dim)
        else:
            cat_emb = self.cat_embedding(cat_ids.view(-1))
        
        if pos_price is not None:
            expert_price = pos_price
        else:
            expert_price = price.view(-1, 1)
        
        # 4. 仅对被激活的TopN个专家执行前向计算
        s_price = torch.zeros(scenario_flat.size(0), device=scenario_flat.device)
        
        for n in range(self.top_n):
            # 获取第n个激活的专家索引
            expert_idx = top_indices[:, n]  # (batch_size * maxlen,)
            gate_weight = alpha[:, n]  # (batch_size * maxlen,) 注意：这里需要取对应位置的值
            
            # 获取实际激活的专家
            unique_experts = torch.unique(expert_idx)
            
            for expert_id in unique_experts:
                # 找到使用该专家的样本
                mask = (expert_idx == expert_id)
                if mask.sum() == 0:
                    continue
                
                # 执行专家前向计算
                expert_output = self.experts[expert_id](
                    expert_price[mask],
                    cat_emb[mask]
                ).squeeze(-1)  # (num_activated,)
                
                # 加权累加
                s_price[mask] += gate_weight[mask] * expert_output
        
        return s_price, alpha, top_indices
    
    def compute_reg_loss(self, alpha):
        """
        计算防坍缩正则项
        L_reg = -λ × Σ α × log(α)
        """
        # 添加小epsilon防止log(0)
        alpha_safe = torch.clamp(alpha, min=1e-8)
        
        # 计算熵正则项: -Σ α × log(α)
        reg_loss = -torch.sum(alpha * torch.log(alpha_safe), dim=-1)
        
        return reg_loss.mean()
