# -*- coding: utf-8 -*-
"""
PyTorch implementation of Transformer modules for SASRec
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionalEncoding(nn.Module):
    """Positional Encoding module"""
    
    def __init__(self, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, dim)
        Returns:
            Positional encoding of shape (batch_size, seq_len, dim)
        """
        return self.pe[:x.size(1), :].unsqueeze(0).expand(x.size(0), -1, -1)


class LayerNorm(nn.Module):
    """Layer Normalization"""
    
    def __init__(self, features, eps=1e-8):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""
    
    def __init__(self, num_units, num_heads=8, dropout_rate=0.0, causality=False):
        super(MultiHeadAttention, self).__init__()
        
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        
        assert num_units % num_heads == 0, "num_units must be divisible by num_heads"
        self.depth = num_units // num_heads
        
        # Linear projections for Q, K, V
        self.W_query = nn.Linear(num_units, num_units)
        self.W_key = nn.Linear(num_units, num_units)
        self.W_value = nn.Linear(num_units, num_units)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = LayerNorm(num_units)
    
    def forward(self, queries, keys, mask=None):
        """
        Args:
            queries: (batch_size, seq_len_q, num_units)
            keys: (batch_size, seq_len_k, num_units)
            mask: (batch_size, seq_len_q, seq_len_k) or None
        Returns:
            outputs: (batch_size, seq_len_q, num_units)
        """
        batch_size = queries.size(0)
        
        # Linear projections and split into heads
        Q = self.W_query(queries).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        K = self.W_key(keys).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        V = self.W_value(keys).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.depth)
        
        # Apply mask if provided (for padding)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Causality mask (for autoregressive property)
        if self.causality:
            seq_len_q = queries.size(1)
            seq_len_k = keys.size(1)
            causal_mask = torch.tril(torch.ones(seq_len_q, seq_len_k, device=queries.device)).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(causal_mask == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum
        outputs = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        outputs = outputs.transpose(1, 2).contiguous().view(batch_size, -1, self.num_units)
        
        # Residual connection and layer normalization
        outputs = self.layer_norm(outputs + queries)
        
        return outputs


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    
    def __init__(self, num_units, ff_units, dropout_rate=0.0):
        super(FeedForward, self).__init__()
        
        self.fc1 = nn.Linear(num_units, ff_units[0])
        self.fc2 = nn.Linear(ff_units[0], ff_units[1] if len(ff_units) > 1 else num_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = LayerNorm(num_units if len(ff_units) == 1 else ff_units[1])
        self.num_units = num_units
        self.output_dim = ff_units[1] if len(ff_units) > 1 else num_units
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, num_units)
        Returns:
            outputs: (batch_size, seq_len, output_dim)
        """
        residual = x
        outputs = F.relu(self.fc1(x))
        outputs = self.dropout(outputs)
        outputs = self.fc2(outputs)
        outputs = self.dropout(outputs)
        
        # Residual connection (if dimensions match)
        if self.output_dim == self.num_units:
            outputs = self.layer_norm(outputs + residual)
        else:
            outputs = self.layer_norm(outputs)
        
        return outputs


class Embedding(nn.Module):
    """Embedding layer with optional zero padding and scaling"""
    
    def __init__(self, vocab_size, num_units, zero_pad=True, scale=True, padding_idx=0):
        super(Embedding, self).__init__()
        
        self.num_units = num_units
        self.scale = scale
        self.zero_pad = zero_pad
        
        self.embedding = nn.Embedding(vocab_size, num_units, padding_idx=padding_idx if zero_pad else None)
        
        # Initialize with Xavier uniform
        nn.init.xavier_uniform_(self.embedding.weight)
        
        if zero_pad:
            with torch.no_grad():
                self.embedding.weight[padding_idx].fill_(0)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len) of token indices
        Returns:
            embeddings: (batch_size, seq_len, num_units)
        """
        output = self.embedding(x)
        if self.scale:
            output = output * np.sqrt(self.num_units)
        return output
