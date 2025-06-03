'''
Date: 2025-04-11 11:59:04
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-06-03 10:55:45
Description: 
'''
import platform
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from mbapy.dl_torch.bb import RoPE

SERVER = platform.uname().node
ROOT = Path(f'/home/{SERVER}/Desktop/BHM/CB1-Pepcans-MDS/')

import sys

sys.path.append(str(ROOT / 'MuMoPepcan'))
from model.functional import add_rope

# remain for compatibility
class Attn(nn.Module):
    """TransformerEncoder with pred-token, input is [N, L, feat_dim], output is [N, 1, feat_dim].
    Args:
        feat_dim (int): Feature dimension of the input.
        num_layers (int): Number of transformer encoder layers.
        nhead (int): Number of attention heads.
        dropout (float): Dropout rate for the transformer encoder.
        
    forward method:
        feat (torch.Tensor): Input tensor of shape [N, L, feat_dim].
        mask (torch.Tensor): Optional attention mask of shape [N, L], False means padding.
        Returns:
            torch.Tensor: Output tensor of shape [N, 1, feat_dim].
    """
    def __init__(self, feat_dim: int = 384, num_layers: int = 2, nhead: int = 8, dropout: float = 0.4):
        super().__init__()
        # Learnable prediction token
        self.pred_token = nn.Parameter(torch.zeros(1, 1, feat_dim))  # [1, 1, 384]
        nn.init.xavier_uniform_(self.pred_token)  # Initialize with Xavier uniform
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=feat_dim, nhead=nhead,
                                                   dim_feedforward=4*feat_dim,
                                                   dropout=dropout, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, feat: torch.Tensor, mask: torch.Tensor):
        batch_size, L, feat_dim = feat.shape  # [N, L, 384]
        
        # Add learnable prediction token
        pred_token = self.pred_token.expand(batch_size, -1, -1)  # [N, 1, 384]
        smile_feat = torch.cat([pred_token, feat], dim=1)  # [N, L+1, 384]
        
        # Add RoPE (Rotary Position Embedding)
        smile_feat = add_rope(smile_feat)  # [N, L+1, 384]
        
        # Create attention mask for Transformer
        if mask is not None:
            extended_mask = torch.cat([torch.ones(batch_size, 1, device=mask.device), mask], dim=1)  # [N, L+1]
            transformer_mask = ~extended_mask.bool()  # [N, L+1]
        else:
            transformer_mask = None
        
        # Pass through Transformer
        transformer_out = self.transformer(smile_feat.transpose(0, 1), src_key_padding_mask=transformer_mask)  # [L+1, N, 384]
        transformer_out = transformer_out.transpose(0, 1)  # [N, L+1, 384]
        
        # Extract prediction token output
        return transformer_out[:, 0:1, :]  # [N, 1, 384]


class PredTokenAttn(nn.Module):
    def __init__(self, n_token: int = 1, feat_dim: int = 384, n_layer: int = 2, n_head: int = 8, dropout: float = 0.4):
        """TransformerEncoder with pred-token, input is [N, L, feat_dim], output is [N, n_token, feat_dim].
        Args:
            n_token (int): Number of prediction tokens.
            feat_dim (int): Feature dimension of the input.
            n_layer (int): Number of transformer encoder layers.
            n_head (int): Number of attention heads.
            dropout (float): Dropout rate for the transformer encoder.
            
        forward method:
            feat (torch.Tensor): Input tensor of shape [N, L, feat_dim].
            mask (torch.Tensor): Optional attention mask of shape [N, L], False means padding.
            Returns:
                torch.Tensor: Output tensor of shape [N, n_token, feat_dim].
        """
        super().__init__()
        # Learnable prediction token
        self.n_token = n_token
        self.pred_token = nn.Parameter(torch.zeros(1, n_token, feat_dim))  # [1, n_token, 384]
        nn.init.xavier_uniform_(self.pred_token)  # Initialize with Xavier uniform
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=feat_dim, nhead=n_head,
                                                   dim_feedforward=4*feat_dim,
                                                   dropout=dropout, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
    def forward(self, feat: torch.Tensor, mask: torch.Tensor):
        batch_size, L, feat_dim = feat.shape  # [N, L, 384]
        
        # Add learnable prediction token
        pred_token = self.pred_token.expand(batch_size, -1, -1)  # [N, n_token, 384]
        smile_feat = torch.cat([pred_token, feat], dim=1)  # [N, n_token+L, 384]
        
        # Add RoPE (Rotary Position Embedding)
        smile_feat = add_rope(smile_feat)  # [N, n_token+L, 384]
        
        # Create attention mask for Transformer
        if mask is not None:
            extended_mask = torch.cat([torch.ones(batch_size, self.n_token, device=mask.device), mask], dim=1)  # [N, n_token+L]
            transformer_mask = ~extended_mask.bool()  # [N, n_token+L]
        else:
            transformer_mask = None
        
        # Pass through Transformer
        transformer_out = self.transformer(smile_feat.transpose(0, 1), src_key_padding_mask=transformer_mask)  # [n_token+L, N, 384]
        transformer_out = transformer_out.transpose(0, 1)  # [N, n_token+L, 384]
        
        # Extract prediction token output
        return transformer_out[:, :self.n_token, :]  # [N, n_token, 384]
    

class CrossAttn(nn.Module):
    """Cross Attention Module, input is [N, L1, D] and [N, L2, D], output is [N, L1, D] and [N, L1, L2].
    Args:
        feat_dim (int): Feature dimension of the input.
        n_layer (int): Number of transformer decoder layers.
        n_head (int): Number of attention heads.
        dropout (float): Dropout rate for the transformer decoder.
        
    forward method:
        q (torch.Tensor): Query tensor of shape [N, L1, D].
        k (torch.Tensor): Key tensor of shape [N, L2, D].
        mask (torch.Tensor): Optional attention mask of shape [N, L1], False means padding.
        Returns:
            torch.Tensor: Output tensor of shape [N, L1, D].
            torch.Tensor: Output tensor of shape [N, L2, D]."""
    def __init__(self, feat_dim: int = 384, n_layer: int = 2, n_head: int = 8, dropout: float = 0.4):
        super().__init__()
        self.rope = RoPE(feat_dim, 1000)
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=n_head, dropout=dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=feat_dim, nhead=n_head,
                                                 dim_feedforward=4*feat_dim,
                                                 dropout=dropout, activation='relu')
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layer)
        
    def get_attn(self, q: torch.Tensor, k: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # [N, L, D] -> [N, L, D]
        q, k = add_rope(q), add_rope(k)
        
        
        _, attn_output_weights = self.cross_attn(
            query=q.transpose(0, 1), 
            key=k.transpose(0, 1), 
            value=k.transpose(0, 1)
        )  # [Lq, N, D]
        
        return attn_output_weights

    def forward(self, q: torch.Tensor, k: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # [N, L, D] -> [N, L, D]
        q, k = add_rope(q), add_rope(k)
        
        
        attn_output, _ = self.cross_attn(
            query=q.transpose(0, 1), 
            key=k.transpose(0, 1), 
            value=k.transpose(0, 1)
        )  # [Lq, N, D]
        
        
        decoder_q = self.transformer(
            tgt=attn_output,
            memory=q.transpose(0, 1),
            tgt_key_padding_mask=mask
        ).transpose(0, 1)  # [N, Lq, D]
        decoder_k = self.transformer(
            tgt=attn_output,
            memory=k.transpose(0, 1),
            tgt_key_padding_mask=mask
        ).transpose(0, 1)  # [N, Lq, D]
        
        return decoder_q, decoder_k