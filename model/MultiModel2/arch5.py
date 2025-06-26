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
from model.predictor import MCMLP, MHMLP, MLDecoderLite, SimpleMLP


class MultiModel(nn.Module):
    """
    MultiModel2
    -----------
    models: 
        - CNN Encoder: COneD with cnn_cfg and each layer RoPE settings
        - SMILES Encoder: PredTokenAttn(n_token=4)
        - Cross Attntion: CrossAttn
        - wet_predictor: MCMLP
        - plip_redictor: MLDecoderLite(128)
    """
    def __init__(self, args: GlobalSettings, cnn_cfg: list[LayerCfg], hidden_dim: int, out_dim1: int, out_dim2: int,
                 n_layer: int = 1, n_head: int = 8, dropout: float = 0.4,
                 RoPE: list[bool] = None):
        super().__init__()
        self.pose_cnn = CNN1D(args, cnn_cfg)
        RoPE = RoPE or [False for _ in range(len(self.pose_cnn.cnn.main_layers))]
        for i, rope in enumerate(RoPE):
            if rope:
                self.pose_cnn.cnn.main_layers[i].trans[0].self_attention.RoPE = _RoPE(cnn_cfg[i].outc, 1000)
        self.smiles_attn = PredTokenAttn(n_token=4, feat_dim=hidden_dim, n_layer=n_layer, n_head=n_head, dropout=dropout)
        self.cross_attn = CrossAttn(feat_dim=hidden_dim, n_layer=n_layer, n_head=n_head, dropout=dropout)
        # multi-predictor for wet
        self.wet_predictor = MCMLP(num_classes=out_dim1, input_dim=hidden_dim, dropout=dropout)
        # MPL-predictor for PLIP
        self.plip_redictor = MLDecoderLite(out_dim2, hidden_dim, n_head=n_head)
        
    def forward(self, pose: torch.Tensor, smiles: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Pose: [N, L, 3] -> [N, 3, L] -pose_cnn> [N, D, L]
        pose_feat = self.pose_cnn(pose.permute(0, 2, 1))
        # SMILES: [N, L2, D] -smiles_attn> [N, 1, D]
        smiles_feat = self.smiles_attn(smiles, mask)
        # cross-attention for pose and smiles: -> [N, 1, D]
        aligned_smiles_feat, aligned_pose_feat = self.cross_attn(smiles_feat, pose_feat.permute(0, 2, 1))
        # multi-task predict
        wet_p = self.wet_predictor(aligned_smiles_feat).squeeze(1) # [N, out_dim]
        plip_p = self.plip_redictor(aligned_pose_feat) # [N, out_dim]
        return wet_p, plip_p