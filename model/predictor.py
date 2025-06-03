'''
Date: 2025-04-11 10:43:06
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-04-25 16:25:03
Description: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from mbapy.dl_torch.bb import EncoderLayer, PositionwiseFeedforwardLayer


class SimpleMLP(nn.Module):
    """Simple 2-layer MLP
    Args:
        num_classes (int): Number of classes.
        input_dim (int): Input dimension.
        dropout (float, optional): Dropout rate. Defaults to 0.4.
    """
    def __init__(self, input_dim: int, out_dim: int = 1, dropout: float = 0.4):
        super().__init__()
        # Fully connected layers for output
        self.out_fc = nn.Sequential(nn.Linear(input_dim, 4 * input_dim),
                                    nn.LeakyReLU(),
                                    nn.Dropout(dropout),  # 添加 Dropout
                                    nn.Linear(4 * input_dim, out_dim),
                                    nn.LeakyReLU()
                                    )
    
    def forward(self, x):
        # x: [b, C]
        return self.out_fc(x)  # [N, out_dim]


class MHMLP(nn.Module):
    """Multi head MLP
    Args:
        num_classes (int): Number of classes.
        input_dim (int): Input dimension.
        dropout (float, optional): Dropout rate. Defaults to 0.4.
    """
    def __init__(self, num_classes: int, input_dim: int, dropout: float = 0.4):
        super().__init__()
        # Fully connected layers for output
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.out_fcs = nn.ModuleList([SimpleMLP(input_dim, 1, dropout) for _ in range(num_classes)])
    
    def forward(self, x):
        # x: [b, C]
        return torch.cat([fc_i(x) for fc_i in self.out_fcs], dim=-1)  # [N, out_dim]


class MCMLP(nn.Module):
    """Multi channle MLP
    Args:
        num_classes (int): Number of classes.
        input_dim (int): Input dimension.
        dropout (float, optional): Dropout rate. Defaults to 0.4.
    """
    def __init__(self, num_classes: int, input_dim: int, dropout: float = 0.4):
        super().__init__()
        # Fully connected layers for output
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.out_fcs = nn.ModuleList([SimpleMLP(input_dim, 1, dropout) for _ in range(num_classes)])
    
    def forward(self, x):
        # x: [b, NC, D] -> [b, NC]
        return torch.cat([self.out_fcs[i](x[:, i, :]) for i in range(self.num_classes)], dim=-1)


class MLDecoderLite(nn.Module):
    def __init__(self, num_classes: int, input_dim: int, n_head: int = 8, dropout: float = 0.3):
        """MLDecoderLite
        Args:
            num_classes (int): Number of classes.
            input_dim (int): Input dimension.
            n_heads (int, optional): Number of attention heads. Defaults to 8.
            
        forward:
            x (torch.Tensor): Input tensor of shape [b, L, C].
            return (torch.Tensor): Output tensor of shape [b, NC].
        """
        super(MLDecoderLite, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.head_dim = input_dim // n_head
        self.n_head = n_head
        # ML_Decoder : CrossAttention
        self.gq = nn.Parameter(torch.zeros(1, num_classes, input_dim))
        torch.nn.init.xavier_uniform_(self.gq)
        self.cross_attn = nn.MultiheadAttention(input_dim, n_head, batch_first=True)
        # ML_Decoder : FFN
        self.self_attn_layer_norm = nn.LayerNorm(input_dim)
        self.ff_layer_norm = nn.LayerNorm(input_dim)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(input_dim, 4*input_dim, dropout)
        self.dropout_FFN = nn.Dropout(dropout)
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