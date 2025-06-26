'''
Date: 2025-04-13 21:38:52
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-06-26 22:14:57
Description: 
'''
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from mbapy.dl_torch.bb import RoPE as _RoPE
from mbapy.dl_torch.m import LayerCfg, TransCfg
from mbapy.dl_torch.utils import GlobalSettings

SERVER = platform.uname().node
ROOT = Path(__file__).parent.parent.parent.parent

import sys

sys.path.append(str(ROOT / 'MuMoPepcan'))

from model.CNN.cnn import CNN1D, CNN2D
from model.MHSA.attn import CrossAttn, PredTokenAttn
from model.predictor import MHMLP, SimpleMLP


class MultiModel(nn.Module):
    """
    MultiModel2
    -----------
    models: 
        - CNN Encoder: COneD with cnn_cfg and each layer RoPE settings
        - SMILES Encoder: PredTokenAttn(n_token=1)
        - Cross Attntion: CrossAttn
        - wet_predictor: MHMLP
        - plip_redictor: SimpleMLP
    """
    def __init__(self, args: GlobalSettings, cnn_cfg: list[LayerCfg], hidden_dim: int, out_dim1: int, out_dim2: int,
                 n_layer: int = 1, n_head: int = 8, dropout: float = 0.4,
                 RoPE: list[bool] = None):
        super().__init__()
        # CNN Encoder
        self.pose_cnn = CNN1D(args, cnn_cfg)
        RoPE = RoPE or [False for _ in range(len(self.pose_cnn.cnn.main_layers))]
        for i, rope in enumerate(RoPE):
            if rope:
                self.pose_cnn.cnn.main_layers[i].trans[0].self_attention.RoPE = _RoPE(cnn_cfg[i].outc, 1000)
        # SMILES Encoder
        self.smiles_attn = PredTokenAttn(feat_dim=hidden_dim, n_layer=n_layer, n_head=n_head, dropout=dropout)
        # Cross Attntion
        self.cross_attn = CrossAttn(feat_dim=hidden_dim, n_layer=n_layer, n_head=n_head, dropout=dropout)
        # multi-predictor for wet
        self.wet_predictor = MHMLP(num_classes=out_dim1, input_dim=hidden_dim, dropout=dropout)
        # MPL-predictor for PLIP
        self.plip_redictor = SimpleMLP(hidden_dim, out_dim2, dropout)
        
    def get_attn(self, pose: torch.Tensor, smiles: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Pose: [N, L, 3] -> [N, 3, L] -pose_cnn> [N, D, L]
        pose_feat = self.pose_cnn(pose.permute(0, 2, 1))
        smiles_feat = self.smiles_attn(smiles, mask)
        return self.cross_attn.get_attn(smiles_feat, pose_feat.permute(0, 2, 1))       
        
    def forward(self, pose: torch.Tensor, smiles: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Pose: [N, L, 3] -> [N, 3, L] -pose_cnn> [N, D, L]
        pose_feat = self.pose_cnn(pose.permute(0, 2, 1))
        # SMILES: [N, L2, D] -smiles_attn> [N, 1, D]
        smiles_feat = self.smiles_attn(smiles, mask)
        # cross-attention for pose and smiles: -> [N, 1, D]
        aligned_smiles_feat, aligned_pose_feat = self.cross_attn(smiles_feat, pose_feat.permute(0, 2, 1))
        # multi-task predict
        wet_p = self.wet_predictor(aligned_smiles_feat).squeeze(1) # [N, out_dim]
        plip_p = self.plip_redictor(aligned_pose_feat).squeeze(1) # [N, out_dim]
        return wet_p, plip_p


if __name__ == '__main__':
    # dev code
    import tempfile
    from mbapy.dl_torch.utils import Mprint
    import hiddenlayer as hl
    import tensorwatch as tw
    from torchviz import make_dot
    # create model
    hidden_dim, dropout = 384, 0.4
    cfg = [
        LayerCfg( 3,   8, 32, 1, 'SABlock1D', avg_size=1, use_trans=False), # 1212 -> 1212
        LayerCfg( 8,  16, 16, 1, 'SABlock1D', avg_size=4, use_trans=False), # 1212 -> 303
        LayerCfg(16,  32, 16, 1, 'SABlock1D', avg_size=1, use_trans=True, # 303 -> 303
                trans_layer='EncoderLayer', trans_cfg=TransCfg(32, n_layers=1, dropout=dropout)),
        LayerCfg(32, hidden_dim,  3, 1, 'SABlock1D', avg_size=1, use_trans=True, # 303 -> 303
                trans_layer='EncoderLayer', trans_cfg=TransCfg(384, n_layers=1, dropout=dropout)),
        ]
    with tempfile.NamedTemporaryFile() as temp_file:
        args = GlobalSettings(Mprint(temp_file.name), temp_file.name)
    model = MultiModel(args, cfg, hidden_dim, 3, 1).cuda()
    # torch.onnx.export(model,                     # 模型
    #              (torch.randn(2, 932, 3).cuda(), torch.randn(2, 64, 384).cuda(), torch.randn(2, 64).cuda()),  # 示例输入
    #              "model.onnx",               # 输出文件名
    #              export_params=True)
    
    # y = model(torch.randn(2, 932, 3).cuda(), torch.randn(2, 64, 384).cuda(), torch.randn(2, 64).cuda())
    # make_dot(y, params=dict(model.named_parameters())).render("model_viz", format="png")
    
    # graph = hl.build_graph(model, (torch.randn(2, 932, 3).cuda(), torch.randn(2, 64, 384).cuda(), torch.randn(2, 64).cuda()))
    # graph.theme = hl.graph.THEMES["blue"].copy()
    # graph.save("model_architecture", format="png")
    
    # tw.draw_model(model, ([2, 932, 3], [2, 64, 384], [2, 64]), png_filename="model_architecture_tw.png")