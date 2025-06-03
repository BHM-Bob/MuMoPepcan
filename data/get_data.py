'''
Date: 2025-04-04 10:59:36
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-06-03 15:06:30
Description: 
'''
import platform
from pathlib import Path

import pandas as pd
import torch
from mbapy.base import put_log
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SERVER = platform.uname().node
ROOT = Path(f'/home/{SERVER}/Desktop/BHM/CB1-Pepcans-MDS/')

def load_pose_data(pose_type: str = 'aligned_TM_pos', verbose: bool = True,
                   frame_start: int = 1, frame_end: int = 10001, frame_step: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    加载pose数据
    '''
    pose = torch.load(str(ROOT / f'MuMoPepcan/data/processed_data/{pose_type}.pt'))
    path_df = pd.read_excel(ROOT / 'MuMoPepcan/data/paths.xlsx', sheet_name='final')
    train_pos, valid_pos = [], []
    for name, t in tqdm(zip(path_df['name'], path_df['t']), desc='Loading Coords', total=len(path_df), disable=not verbose):
        if name == 'single-receptor':
            continue
        if name.startswith('3'):
            valid_pos.append(pose[name][t][frame_start:frame_end:frame_step].clone())
        else:
            train_pos.append(pose[name][t][frame_start:frame_end:frame_step].clone())
        del pose[name][t]
    data_train = torch.cat(train_pos, dim=0)
    data_valid = torch.cat(valid_pos, dim=0)
    return data_train, data_valid


def load_PLIP_one_hot(verbose: bool = True,
                      frame_start: int = 0, frame_end: int = 10000, frame_step: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    加载PLIP数据, original PLIP analysis starts from frame 1, so we starts 0 as global 1.
    '''
    path_df = pd.read_excel(ROOT / 'MuMoPepcan/data/paths.xlsx', sheet_name='final')
    plip = torch.load(str(ROOT / 'MuMoPepcan/data/processed_data/MDS_plip_interactions_one_hot.pt'))
    train_plip, valid_plip = [], []
    for name, t in tqdm(zip(path_df['name'], path_df['t']), desc='Loading PLIP', total=len(path_df), disable=not verbose):
        if name == 'single-receptor':
            continue
        if name.startswith('3'):
            valid_plip.append(plip[name][t][frame_start:frame_end:frame_step].clone())
        else:
            train_plip.append(plip[name][t][frame_start:frame_end:frame_step].clone())
        del plip[name][t]
    data_train = torch.cat(train_plip, dim=0)
    data_valid = torch.cat(valid_plip, dim=0)
    return data_train, data_valid


def load_SMILES_feat(feat_name: str = 'PepDoRA', verbose: bool = True,
                     frame_start: int = 0, frame_end: int = 10000, frame_step: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    加载SMILES特征
    '''
    n_repeat = (frame_end - frame_start) // frame_step
    path_df = pd.read_excel(ROOT / 'MuMoPepcan/data/paths.xlsx', sheet_name='final')
    smiles = torch.load(str(ROOT / f'MuMoPepcan/data/processed_data/ligand_feats_{feat_name}.pt'))
    feat, mask = smiles['hidden_states'], smiles['attention_mask']
    train_idx, valid_idx = [], []
    for name, t in tqdm(zip(path_df['name'], path_df['t']), desc='Loading SMILES', total=len(path_df), disable=not verbose):
        if name == 'single-receptor':
            continue
        if name.startswith('3'):
            valid_idx.append(torch.tensor(smiles['name'].index(name), dtype=torch.int8).repeat(n_repeat))
        else:
            train_idx.append(torch.tensor(smiles['name'].index(name), dtype=torch.int8).repeat(n_repeat))
    train_idx = torch.cat(train_idx, dim=0)
    valid_idx = torch.cat(valid_idx, dim=0)
    return feat, mask, train_idx, valid_idx


def load_wet_exp_label(verbose: bool = True,
                       frame_start: int = 0, frame_end: int = 10000, frame_step: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    加载 wet exp label
    '''
    n_repeat = (frame_end - frame_start) // frame_step
    path_df = pd.read_excel(ROOT / 'MuMoPepcan/data/paths.xlsx', sheet_name='final')
    names = list(path_df.loc[path_df['name']!='single-receptor', 'name'].unique())
    wet_labels = pd.read_excel(str(ROOT / 'MuMoPepcan/data/label/wet_exp.xlsx'), index_col='name')
    wet_labels = torch.tensor(wet_labels.loc[names, ['animal', 'cell_100', 'cell_300', 'cell_1000']].values,
                              dtype=torch.float32)
    train_idx, valid_idx = [], []
    for name, t in tqdm(zip(path_df['name'], path_df['t']), desc='Loading wet exp', total=len(path_df), disable=not verbose):
        if name =='single-receptor':
            continue
        if name.startswith('3'):
            valid_idx.append(torch.tensor(names.index(name), dtype=torch.int8).repeat(n_repeat))
        else:
            train_idx.append(torch.tensor(names.index(name), dtype=torch.int8).repeat(n_repeat))
    train_idx = torch.cat(train_idx, dim=0)
    valid_idx = torch.cat(valid_idx, dim=0)
    return wet_labels, train_idx, valid_idx


class MultiDataset(Dataset):
    """
    """
    def __init__(self, pose: torch.Tensor, smiles: torch.Tensor, smiles_mask: torch.Tensor, smiles_idx: torch.Tensor,
                 plip: torch.Tensor, wet_labels: torch.Tensor, wet_idx: torch.Tensor):
        super(MultiDataset, self).__init__()
        self.pose = pose
        self.plip = plip
        self.smiles = smiles
        self.smiles_mask = smiles_mask
        self.smiles_idx = smiles_idx
        self.wet_labels = wet_labels
        self.wet_idx = wet_idx
        self.size = pose.shape[0]
        
    def __getitem__(self, idx):
        return self.pose[idx], self.smiles[self.smiles_idx[idx]], self.smiles_mask[self.smiles_idx[idx]], self.plip[idx], self.wet_labels[self.wet_idx[idx]]

    def __len__(self):
        return self.size


if __name__ == '__main__':
    smiles = torch.load(str(ROOT / f'MuMoPepcan/data/processed_data/ligand_feats_PepDoRA_seq.pt'))
    put_log('')
    data = load_pose_data()
    put_log('')
    print(data[0].shape, data[1].shape)
