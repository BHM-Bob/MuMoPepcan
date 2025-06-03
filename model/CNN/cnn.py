'''
Date: 2025-04-03 20:59:02
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-04-22 15:30:18
Description: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from mbapy.dl_torch.bb import RoPE
from mbapy.dl_torch.m import (MATTPE, COneD, COneDLayer, LayerCfg, MAvlayer,
                              TransCfg)
from mbapy.dl_torch.utils import GlobalSettings


class CNN1D(nn.Module):
    def __init__(self, args: GlobalSettings, cfg: list[LayerCfg]):
        super().__init__()
        # cfg = [
        #     LayerCfg( 3,   8, 32, 1, 'SABlock1D', avg_size=1, use_trans=False), # 1212 -> 1212
        #     LayerCfg( 8,  16, 16, 1, 'SABlock1D', avg_size=4, use_trans=False), # 1212 -> 303
        #     LayerCfg(16,  32, 16, 1, 'SABlock1D', avg_size=1, use_trans=True, # 303 -> 303
        #             trans_layer='EncoderLayer', trans_cfg=TransCfg(32, n_layers=1, dropout=dropout)),
        #     LayerCfg(32, hidden_dim,  3, 1, 'SABlock1D', avg_size=1, use_trans=True, # 303 -> 303
        #             trans_layer='EncoderLayer', trans_cfg=TransCfg(384, n_layers=1, dropout=dropout)),
        #     ]
        self.cnn = COneD(args, cfg, COneDLayer)
        
    def forward(self, x: torch.Tensor):
        x = self.cnn(x)
        return x


class CNN2D(nn.Module):
    def __init__(self, args: GlobalSettings, cfg: list[LayerCfg], padding: int, hidden_dim: int):
        """This class implements a 2D Convolutional Neural Network for 1D data, input and output dim is same as 1D CNN.
        Parameters:
        ----------
            args: GlobalSettings, The global settings for the model.
            padding: int, The padding size for the input data, L+padding must be a perfect square.
            hidden_dim: int, The hidden dimension of the model.
            dropout: float, The dropout rate for the model.
            
        forward:
        ----------
            x: torch.Tensor, The input data, shape (batch_size, C, L).
                L is the length of the input data, C is the number of channels.
                The input data will be padded to (batch_size, L+padding, C).
                L+padding must be a perfect square.
        Returns:
        ----------
            x: torch.Tensor, The output data, shape (batch_size, hidden_dim, L').
        """
        super().__init__()
        # cfg = [
        #     LayerCfg( 3,   8,  8, 1, 'SABlockR', avg_size=2, use_trans=False), # 1212 -> 1212
        #     LayerCfg( 8,  16,  8, 1, 'SABlockR', avg_size=2, use_trans=False), # 1212 -> 303
        #     LayerCfg(16,  32, 16, 1, 'SABlockR', avg_size=1, use_trans=True, # 303 -> 303
        #             trans_layer='EncoderLayer', trans_cfg=TransCfg(32, n_layers=1, dropout=dropout)),
        #     LayerCfg(32, hidden_dim,  3, 1, 'SABlockR', avg_size=1, use_trans=True, # 303 -> 303
        #             trans_layer='EncoderLayer', trans_cfg=TransCfg(384, n_layers=1, dropout=dropout)),
        #     ]
        self.padding = padding
        self.hidden_dim = hidden_dim
        self.cnn = MATTPE(args, cfg, MAvlayer)
        
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)
        # x: [b, L, C] -> [b, L+padding, C]
        # padding: 1D padding, not 2D padding
        x = F.pad(x, (0, 0, 0, self.padding), mode='constant', value=0)
        batch_size, L, C = x.size()
        img_size = int(L**0.5)
        x = x.view(batch_size, C, img_size, img_size)
        x = self.cnn(x)
        return x.reshape(batch_size, -1, self.hidden_dim).permute(0, 2, 1)
    

if __name__ == '__main__':
    # dev code
    import tempfile

    from mbapy.dl_torch.utils import Mprint
    x = torch.randn(2, 932, 3).cuda()
    with tempfile.NamedTemporaryFile() as temp_file:
        print(CNN2D(GlobalSettings(Mprint(temp_file.name), temp_file.name), 1024-932, 384).cuda()(x).shape)
    