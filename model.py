import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Embedding, MultiHeadAttention, FeedForward, LayerNorm, PositionalEncoding
from modules.moe_modules import MoEPriceTower


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation Model (PyTorch Version)"""
    
    def __init__(self, usernum, itemnum, args):
        super(SASRec, self).__init__()
        
        self.usernum = usernum
        self.itemnum = itemnum
        self.args = args
        
        # Item embedding table
        self.item_emb = Embedding(
            vocab_size=itemnum + 1,
            num_units=args.hidden_units,
            zero_pad=True,
            scale=True,
            padding_idx=0
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(args.hidden_units, max_len=args.maxlen)
        
        # Dropout
        self.dropout = nn.Dropout(args.dropout_rate)
        
        # Transformer blocks
        self.attention_layers = nn.ModuleList()
        self.feedforward_layers = nn.ModuleList()
        
        for i in range(args.num_blocks):
            self.attention_layers.append(
                MultiHeadAttention(
                    num_units=args.hidden_units,
                    num_heads=args.num_heads,
                    dropout_rate=args.dropout_rate,
                    causality=True
                )
            )
            self.feedforward_layers.append(
                FeedForward(
                    num_units=args.hidden_units,
                    ff_units=[args.hidden_units, args.hidden_units],
                    dropout_rate=args.dropout_rate
                )
            )
        
        # Final layer normalization
        self.final_norm = LayerNorm(args.hidden_units)
        
        # Loss function
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, input_seq, pos, neg, is_training=True):
        """
        Args:
            input_seq: (batch_size, maxlen) - input sequence
            pos: (batch_size, maxlen) - positive items
            neg: (batch_size, maxlen) - negative items
            is_training: bool
        Returns:
            loss, auc
        """
        batch_size = input_seq.size(0)
        
        # Create padding mask (1 for valid positions, 0 for padding)
        mask = (input_seq != 0).float().unsqueeze(-1)  # (batch_size, maxlen, 1)
        
        # Sequence embedding
        seq = self.item_emb(input_seq)  # (batch_size, maxlen, hidden_units)
        
        # Positional encoding
        pos_enc = self.pos_encoding(seq)  # (batch_size, maxlen, hidden_units)
        seq = seq + pos_enc
        
        # Dropout
        seq = self.dropout(seq) if is_training else seq
        seq = seq * mask
        
        # Transformer blocks
        for i in range(self.args.num_blocks):
            # Self-attention
            seq = self.attention_layers[i](seq, seq)
            seq = seq * mask
            
            # Feed forward
            seq = self.feedforward_layers[i](seq)
            seq = seq * mask
        
        seq = self.final_norm(seq)
        
        # Get embeddings for loss computation
        # Reshape for batch matrix operations
        pos_flat = pos.view(-1)  # (batch_size * maxlen)
        neg_flat = neg.view(-1)  # (batch_size * maxlen)
        seq_flat = seq.view(-1, self.args.hidden_units)  # (batch_size * maxlen, hidden_units)
        
        # Look up embeddings
        pos_emb = self.item_emb.embedding(pos_flat)  # (batch_size * maxlen, hidden_units)
        neg_emb = self.item_emb.embedding(neg_flat)  # (batch_size * maxlen, hidden_units)
        
        # Compute logits
        pos_logits = torch.sum(pos_emb * seq_flat, dim=-1)  # (batch_size * maxlen)
        neg_logits = torch.sum(neg_emb * seq_flat, dim=-1)  # (batch_size * maxlen)
        
        # Create istarget mask
        istarget = (pos_flat != 0).float()
        
        # Compute loss (BPR-like loss)
        pos_loss = -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
        neg_loss = -torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        loss = torch.sum(pos_loss + neg_loss) / torch.sum(istarget)
        
        # Compute AUC
        auc = torch.sum(
            ((torch.sign(pos_logits - neg_logits) + 1) / 2) * istarget
        ) / torch.sum(istarget)
        
        return loss, auc
    
    def predict(self, input_seq, item_idx):
        """
        Args:
            input_seq: (1, maxlen) - input sequence for a single user
            item_idx: (101,) - candidate items (1 positive + 100 negative)
        Returns:
            logits: (101,) - prediction scores
        """
        with torch.no_grad():
            # Create padding mask
            mask = (input_seq != 0).float().unsqueeze(-1)  # (1, maxlen, 1)
            
            # Sequence embedding
            seq = self.item_emb(input_seq)  # (1, maxlen, hidden_units)
            
            # Positional encoding
            pos_enc = self.pos_encoding(seq)
            seq = seq + pos_enc
            
            # Dropout (disabled during inference)
            seq = seq * mask
            
            # Transformer blocks
            for i in range(self.args.num_blocks):
                seq = self.attention_layers[i](seq, seq)
                seq = seq * mask
                seq = self.feedforward_layers[i](seq)
                seq = seq * mask
            
            seq = self.final_norm(seq)
            
            # Get the last position's representation
            seq_last = seq[:, -1, :]  # (1, hidden_units)
            
            # Look up item embeddings
            item_emb = self.item_emb.embedding(item_idx)  # (101, hidden_units)
            
            # Compute logits
            logits = torch.matmul(seq_last, item_emb.t())  # (1, 101)
            
            return logits.squeeze(0)  # (101,)


class SASRecMoE(nn.Module):
    """
    SASRec with MoE Price Tower
    双塔架构：
    - 兴趣塔：原始SASRec，输出兴趣得分 s_interest
    - 价格塔：MoE价格感知模块，输出价格账户得分 s_price
    - 总得分：s = s_interest + beta * s_price
    """
    
    def __init__(self, usernum, itemnum, cat_num, args):
        super(SASRecMoE, self).__init__()
        
        self.usernum = usernum
        self.itemnum = itemnum
        self.cat_num = cat_num
        self.args = args
        
        # ========== 兴趣塔（原始SASRec） ==========
        self.item_emb = Embedding(
            vocab_size=itemnum + 1,
            num_units=args.hidden_units,
            zero_pad=True,
            scale=True,
            padding_idx=0
        )
        
        self.pos_encoding = PositionalEncoding(args.hidden_units, max_len=args.maxlen)
        self.dropout = nn.Dropout(args.dropout_rate)
        
        self.attention_layers = nn.ModuleList()
        self.feedforward_layers = nn.ModuleList()
        
        for i in range(args.num_blocks):
            self.attention_layers.append(
                MultiHeadAttention(
                    num_units=args.hidden_units,
                    num_heads=args.num_heads,
                    dropout_rate=args.dropout_rate,
                    causality=True
                )
            )
            self.feedforward_layers.append(
                FeedForward(
                    num_units=args.hidden_units,
                    ff_units=[args.hidden_units, args.hidden_units],
                    dropout_rate=args.dropout_rate
                )
            )
        
        self.final_norm = LayerNorm(args.hidden_units)
        
        # ========== 价格塔（MoE） ==========
        num_accounts = getattr(args, 'num_accounts', 8)
        top_n = getattr(args, 'top_n', 2)
        temperature = getattr(args, 'temperature', 1.0)
        cat_emb_dim = getattr(args, 'cat_emb_dim', 32)
        
        self.price_tower = MoEPriceTower(
            hidden_units=args.hidden_units,
            cat_num=cat_num,
            num_accounts=num_accounts,
            top_n=top_n,
            temperature=temperature,
            cat_emb_dim=cat_emb_dim
        )
        
        # ========== 双塔融合参数 ==========
        # beta：价格塔权重，可学习参数
        self.beta = nn.Parameter(torch.tensor(
            getattr(args, 'beta', 0.5), dtype=torch.float32
        ))
        
        # 防坍缩正则化系数
        self.lambda_reg = getattr(args, 'lambda_reg', 0.01)
    
    def compute_cat_deviation(self, cat_ids, seq_cat_ids, is_training=True):
        """
        在线计算品类偏离度（严格防止数据泄露）
        公式：品类偏离度 = 1 - cosine(当前品类嵌入, 历史序列品类嵌入的平均池化)
        
        参数:
            cat_ids: (batch_size, maxlen) - 当前物品的品类ID
            seq_cat_ids: (batch_size, maxlen) - 输入序列的品类ID
            is_training: bool
        返回:
            cat_deviation: (batch_size, maxlen) - 品类偏离度
        """
        batch_size, maxlen = cat_ids.shape
        
        # 获取品类嵌入
        cat_emb = self.price_tower.cat_embedding(cat_ids)  # (batch_size, maxlen, cat_emb_dim)
        seq_cat_emb = self.price_tower.cat_embedding(seq_cat_ids)
        
        # 创建掩码（非零位置）
        mask = (seq_cat_ids != 0).unsqueeze(-1).float()  # (batch_size, maxlen, 1)
        
        # 历史品类嵌入的平均池化（防止数据泄露：使用累积平均）
        cat_deviation = torch.zeros(batch_size, maxlen, device=cat_ids.device)
        
        for i in range(maxlen):
            # 只使用当前位置之前的历史数据（防止数据泄露）
            if i == 0:
                # 第一个位置没有历史数据，偏离度为0
                continue
            
            # 历史品类嵌入（前i个位置）
            hist_mask = mask[:, :i, :]  # (batch_size, i, 1)
            hist_cat_emb = seq_cat_emb[:, :i, :]  # (batch_size, i, cat_emb_dim)
            
            # 掩码平均池化
            hist_sum = (hist_cat_emb * hist_mask).sum(dim=1)  # (batch_size, cat_emb_dim)
            hist_count = hist_mask.sum(dim=1).clamp(min=1e-8)  # (batch_size, 1)
            hist_mean = hist_sum / hist_count  # (batch_size, cat_emb_dim)
            
            # 当前品类嵌入
            curr_cat_emb = cat_emb[:, i, :]  # (batch_size, cat_emb_dim)
            
            # 余弦相似度
            hist_mean_norm = F.normalize(hist_mean, dim=-1)
            curr_cat_emb_norm = F.normalize(curr_cat_emb, dim=-1)
            
            cosine_sim = torch.sum(hist_mean_norm * curr_cat_emb_norm, dim=-1)  # (batch_size,)
            
            # 品类偏离度
            cat_deviation[:, i] = 1.0 - cosine_sim
        
        return cat_deviation
    
    def forward(self, input_seq, pos, neg, 
                price_seq=None, price_pos=None, price_neg=None,
                cat_seq=None, cat_pos=None, cat_neg=None,
                price_dev_seq=None, price_dev_pos=None, price_dev_neg=None,
                time_dev_seq=None, time_dev_pos=None, time_dev_neg=None,
                is_training=True):
        """
        参数:
            input_seq: (batch_size, maxlen) - 输入序列
            pos: (batch_size, maxlen) - 正样本
            neg: (batch_size, maxlen) - 负样本
            price_seq/pos/neg: 价格特征
            cat_seq/pos/neg: 品类特征
            price_dev_seq/pos/neg: 价格偏离度特征
            time_dev_seq/pos/neg: 时间偏离度特征
            is_training: bool
        返回:
            loss: 总损失 (L_BPR + L_reg)
            auc: AUC指标
        """
        batch_size = input_seq.size(0)
        maxlen = input_seq.size(1)
        
        # ========== 兴趣塔前向传播 ==========
        mask = (input_seq != 0).float().unsqueeze(-1)
        
        seq = self.item_emb(input_seq)
        pos_enc = self.pos_encoding(seq)
        seq = seq + pos_enc
        seq = self.dropout(seq) if is_training else seq
        seq = seq * mask
        
        for i in range(self.args.num_blocks):
            seq = self.attention_layers[i](seq, seq)
            seq = seq * mask
            seq = self.feedforward_layers[i](seq)
            seq = seq * mask
        
        seq = self.final_norm(seq)
        
        # 展平
        pos_flat = pos.view(-1)
        neg_flat = neg.view(-1)
        seq_flat = seq.view(-1, self.args.hidden_units)
        
        # 兴趣得分
        pos_emb = self.item_emb.embedding(pos_flat)
        neg_emb = self.item_emb.embedding(neg_flat)
        
        s_interest_pos = torch.sum(pos_emb * seq_flat, dim=-1)
        s_interest_neg = torch.sum(neg_emb * seq_flat, dim=-1)
        
        # ========== 价格塔前向传播 ==========
        if price_seq is not None and cat_seq is not None:
            # 在线计算品类偏离度
            cat_dev_seq = self.compute_cat_deviation(cat_seq, cat_seq, is_training)
            cat_dev_pos = self.compute_cat_deviation(cat_pos, cat_seq, is_training)
            cat_dev_neg = self.compute_cat_deviation(cat_neg, cat_seq, is_training)
            
            # 价格塔前向传播
            s_price_seq, alpha_seq, _ = self.price_tower(
                price_seq, price_dev_seq, time_dev_seq, cat_dev_seq, cat_seq
            )
            
            s_price_pos, alpha_pos, _ = self.price_tower(
                price_pos, price_dev_pos, time_dev_pos, cat_dev_pos, cat_pos,
                pos_price=price_pos.view(-1, 1),
                pos_cat_ids=cat_pos.view(-1)
            )
            
            s_price_neg, alpha_neg, _ = self.price_tower(
                price_neg, price_dev_neg, time_dev_neg, cat_dev_neg, cat_neg,
                pos_price=price_neg.view(-1, 1),
                pos_cat_ids=cat_neg.view(-1)
            )
        else:
            # 无特征时的默认值
            s_price_pos = torch.zeros_like(s_interest_pos)
            s_price_neg = torch.zeros_like(s_interest_neg)
            alpha_pos = torch.zeros(batch_size * maxlen, self.price_tower.num_accounts, device=input_seq.device)
            alpha_neg = torch.zeros(batch_size * maxlen, self.price_tower.num_accounts, device=input_seq.device)
        
        # ========== 双塔融合 ==========
        # s = s_interest + beta * s_price
        s_pos = s_interest_pos + self.beta * s_price_pos
        s_neg = s_interest_neg + self.beta * s_price_neg
        
        # ========== BPR损失 ==========
        istarget = (pos_flat != 0).float()
        
        pos_loss = -torch.log(torch.sigmoid(s_pos) + 1e-24) * istarget
        neg_loss = -torch.log(1 - torch.sigmoid(s_neg) + 1e-24) * istarget
        l_bpr = torch.sum(pos_loss + neg_loss) / torch.sum(istarget)
        
        # ========== 防坍缩正则项 ==========
        l_reg = self.price_tower.compute_reg_loss(alpha_pos) + \
                self.price_tower.compute_reg_loss(alpha_neg)
        l_reg = l_reg / 2.0
        
        # ========== 总损失 ==========
        loss = l_bpr + self.lambda_reg * l_reg
        
        # ========== AUC ==========
        auc = torch.sum(
            ((torch.sign(s_pos - s_neg) + 1) / 2) * istarget
        ) / torch.sum(istarget)
        
        return loss, auc
    
    def predict(self, input_seq, item_idx, 
                price_seq=None, cat_seq=None,
                time_dev_seq=None, price_dev_seq=None):
        """
        预测阶段
        参数:
            input_seq: (1, maxlen) - 输入序列
            item_idx: (101,) - 候选物品
            price_seq/cat_seq/time_dev_seq/price_dev_seq: 特征
        返回:
            logits: (101,) - 预测得分
        """
        with torch.no_grad():
            # 兴趣塔
            mask = (input_seq != 0).float().unsqueeze(-1)
            
            seq = self.item_emb(input_seq)
            pos_enc = self.pos_encoding(seq)
            seq = seq + pos_enc
            seq = seq * mask
            
            for i in range(self.args.num_blocks):
                seq = self.attention_layers[i](seq, seq)
                seq = seq * mask
                seq = self.feedforward_layers[i](seq)
                seq = seq * mask
            
            seq = self.final_norm(seq)
            
            seq_last = seq[:, -1, :]  # (1, hidden_units)
            item_emb = self.item_emb.embedding(item_idx)
            
            s_interest = torch.matmul(seq_last, item_emb.t()).squeeze(0)  # (101,)
            
            # 价格塔
            if price_seq is not None and cat_seq is not None:
                # 为每个候选物品计算价格得分
                item_count = item_idx.size(0)
                s_price = torch.zeros(item_count, device=input_seq.device)
                
                for i in range(item_count):
                    item_id = item_idx[i]
                    # 获取该物品的价格、品类等特征（这里简化处理，实际需要从特征映射中获取）
                    # 由于预测阶段可能没有完整的特征，使用默认值
                    s_price[i] = 0.0
            else:
                s_price = torch.zeros_like(s_interest)
            
            # 双塔融合
            logits = s_interest + self.beta * s_price
            
            return logits
