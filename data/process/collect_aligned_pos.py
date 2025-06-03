'''
Date: 2025-04-03 10:18:29
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-04-23 09:58:14
Description: 
'''
import platform
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from lazydock.gmx.mda.align import get_aligned_coords
from lazydock.gmx.mda.utils import filter_atoms_by_chains
from MDAnalysis import AtomGroup, Universe
from tqdm import tqdm

SERVER = platform.uname().node
ROOT = Path(f'/home/{SERVER}/Desktop/BHM/CB1-Pepcans-MDS/')


def get_backbone_atoms(root: Path) -> Tuple[Universe, AtomGroup]:
    tpr_path = str(root/'md.tpr')
    xtc_path = str(root/'md_center.xtc')
    u = Universe(tpr_path, xtc_path)
    # only keep backbone atoms and drop C-Term
    ag = filter_atoms_by_chains(u.select_atoms(f'backbone and resid 0:{413-111+1}'), 'A')
    return u, ag


def get_TM_backbone_atoms(root: Path) -> Tuple[Universe, AtomGroup]:
    gro_path = str(root/'npt.gro')
    xtc_path = str(root/'md_center.xtc')
    u = Universe(gro_path, xtc_path)
    TM_range = [(112, 144+1), (150, 179+1), (185, 220+1), (229, 254+1), (272, 312+1), (332, 369+1), (372, 400+1)]
    TM_ag = u.atoms[np.isin(u.atoms.resids, np.concatenate([np.arange(*r) for r in TM_range])) &\
        np.isin(u.atoms.ids, u.select_atoms('backbone').ids)]
    return u, TM_ag


if __name__ == '__main__':
    path_df = pd.read_excel(ROOT / 'MuMoPepcan/data/paths.xlsx', sheet_name='final')
    coords = {}
    for name, path, t in tqdm(path_df.loc[:, ['name', 'path', 't']].values, total=len(path_df)):
        root = Path(path.replace('Z:\\USERS\\BHM\\LFH\\', '/home/pcmd36/Desktop/BHM/LFH/').replace('\\', '/'))
        u, ag = get_backbone_atoms(root)
        _, aligned_pos = get_aligned_coords(u, ag, 0, 1, 10001, backend='cuda')
        if name not in coords:
            coords[name] = {}
            coords[name][t] = aligned_pos.to(dtype=torch.float32).cpu()
        else:
            coords[name][t] = aligned_pos.to(dtype=torch.float32).cpu()
            
    torch.save(coords, ROOT/'MuMoPepcan/data/processed_data/aligned_pos.pt')