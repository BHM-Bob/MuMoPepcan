'''
Date: 2025-04-11 10:43:06
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-04-11 15:36:20
Description: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from mbapy.dl_torch.bb import EncoderLayer, PositionwiseFeedforwardLayer


@torch.jit.script
class GroupFC(object):
    def __init__(self, embed_len_decoder: int):
        self.embed_len_decoder = embed_len_decoder

    def __call__(self, h: torch.Tensor, duplicate_pooling: torch.Tensor, out_extrap: torch.Tensor):
        for i in range(h.shape[1]):
            h_i = h[:, i, :]
            if len(duplicate_pooling.shape)==3:
                w_i = duplicate_pooling[i, :, :]
            else:
                w_i = duplicate_pooling
            out_extrap[:, i, :] = torch.matmul(h_i, w_i)


class MLDecoderLite(nn.Module):
    def __init__(self, num_classes: int, input_dim: int, n_heads: int = 8):
        super(MLDecoderLite, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.head_dim = input_dim // n_heads
        self.n_heads = n_heads
        # ML_Decoder : CrossAttention
        self.gq = nn.Parameter(torch.zeros(1, num_classes, input_dim))
        torch.nn.init.xavier_uniform_(self.gq)
        self.cross_attn = nn.MultiheadAttention(input_dim, n_heads, batch_first=True)
        # ML_Decoder : FFN
        self.self_attn_layer_norm = nn.LayerNorm(input_dim)
        self.ff_layer_norm = nn.LayerNorm(input_dim)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(input_dim, 4*input_dim, 0.3)
        self.dropout_FFN = nn.Dropout(0.3)
        # ML_Decoder : groupFC
        self.groupFC = nn.Linear(input_dim, 1)  

    def forward(self, x:torch.Tensor)->torch.Tensor:
        # x: [b, L, C]
        batch_size = x.shape[0]
        # gq: [1, NC, C] => [b, NC, C]
        gq = self.gq.repeat(batch_size, 1, 1)
        # _x: [b, NC, C]
        _x = self.cross_attn(gq, x, x)[0]
        # x: [b, NC, C]
        x = self.self_attn_layer_norm(gq + self.dropout_FFN(_x))
        # positionwise feedforward
        _x = self.positionwise_feedforward(x)
        # dropout, residual and layer norm
        x = self.ff_layer_norm(x + self.dropout_FFN(_x))
        # x: [b, NC, C] => [b, NC]
        x = self.groupFC(x).reshape(batch_size, -1)
        return x
    
    
if __name__ == '__main__':
    # dev code
    model = MLDecoderLite(num_classes=4, input_dim=384)
    x = torch.randn(320, 343, 384)
    logits = model(x)
    print(logits.shape)  # should be [320, 4]