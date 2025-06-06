'''
Date: 2025-01-05 20:40:53
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-06-06 19:51:54
Description: 
'''
from glob import glob
import os
import platform
import queue
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from mbapy.base import get_fmt_time
from mbapy.dl_torch.data import DataSetRAM
from mbapy.dl_torch.utils import (AverageMeter, GlobalSettings, Mprint,
                                  ProgressMeter, init_model_parameter,
                                  resume_checkpoint, save_checkpoint)
from mbapy.file import get_paths_with_extension, get_dir
from mbapy.plot import save_show
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from torch.optim import Adam, AdamW

SERVER = platform.uname().node
ROOT = Path(f'/home/{SERVER}/Desktop/BHM/CB1-Pepcans-MDS/')

import sys

sys.path.append(str(ROOT / 'MuMoPepcan'))

from model.MHSA.attn import Attn as AttnBase, PredTokenAttn
from model.MLP.ML_Decoder import MLDecoderLite
from model.predictor import MCMLP


class Attn(nn.Module):
    def __init__(self, hidden_dim: int = 384, out_dim: int = 3, out_dim2: int = 1, num_layers: int = 2, nhead: int = 8, dropout: float = 0.4):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.attn = AttnBase(hidden_dim, num_layers=num_layers, nhead=nhead, dropout=dropout)
        self.predictor = MLDecoderLite(out_dim+out_dim2, hidden_dim, nhead)
        
    def forward(self, smile_feat: torch.Tensor, mask: torch.Tensor):
        # smile_feat: [N, L, 384], mask: [N, L]
        x = self.attn(smile_feat, mask)  # [N, 1, 384]
        x = self.predictor(x)  # [N, 4]
        return x

class Attn2(nn.Module):
    def __init__(self, hidden_dim: int = 384, out_dim: int = 3, out_dim2: int = 1, num_layers: int = 2, nhead: int = 8, dropout: float = 0.4):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Learnable prediction token
        self.pta = PredTokenAttn(out_dim+out_dim2, hidden_dim, n_layer=num_layers, n_head=nhead, dropout=dropout)
        self.out_fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, smile_feat: torch.Tensor, mask: torch.Tensor):
        # Extract prediction token output
        pred_token_out = self.pta(smile_feat, mask)  # [N, 384]
        
        # Pass through fully connected layers
        out = self.out_fc(pred_token_out).squeeze(2)  # [N, out_dim]
        return out

class Attn3(nn.Module):
    def __init__(self, hidden_dim: int = 384, out_dim: int = 3, out_dim2: int = 1, num_layers: int = 2, nhead: int = 8, dropout: float = 0.4):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Learnable prediction token
        self.pta = PredTokenAttn(out_dim+out_dim2, hidden_dim, n_layer=num_layers, n_head=nhead, dropout=dropout)
        self.out_fc = MCMLP(out_dim+out_dim2, hidden_dim, dropout=dropout)
        
    def forward(self, smile_feat: torch.Tensor, mask: torch.Tensor):
        # Extract prediction token output
        pred_token_out = self.pta(smile_feat, mask)  # [N, 4, 384]
        
        # Pass through fully connected layers
        out = self.out_fc(pred_token_out)  # [N, out_dim]
        return out


def train_batch(model: nn.Module, optimizer, criterion: nn.MSELoss, x: torch.Tensor, label: torch.Tensor):
    # add noise
    x[0] += 0.01*torch.randn_like(x[0], device=x[0].device) # [N, ]
    # add random mask
    x[1] = torch.where((x[1] != 0) & torch.rand(x[1].shape, device=x[1].device).gt(0.9), 0, x[1])
    # train
    model.train()
    prs = []
    optimizer.zero_grad()
    out = model(x[0], x[1])
    with torch.no_grad():
        for i in range(label.shape[-1]):
            prs.append(pearsonr(out[:, i].cpu().numpy(), label[:, i].cpu().numpy())[0])
    # like label-smoothing, just add noise
    label[:, 0] += 0.1*torch.randn_like(label[:, 0], device=label.device)
    label[:, 1:] += 0.01*torch.randn_like(label[:, 1:], device=label.device)
    # scale label and out for cell exp
    label[:, 1:] *= 20
    out[:, 1:] *= 20
    # calculate loss and backpropagation
    loss = criterion(out, label)
    loss.backward()
    optimizer.step()
    return loss.item(), np.mean(prs)

def test_batch(model: nn.Module, criterion: nn.MSELoss, x: torch.Tensor, label: torch.Tensor):
    model.eval()
    prs = []
    with torch.no_grad():
        out = model(x[0], x[1])
        loss = criterion(out, label)
        out, label = out.cpu().numpy(), label.cpu().numpy()
        for i in range(label.shape[-1]):
            prs.append(pearsonr(out[:, i], label[:, i])[0])
    return loss.item(), np.mean(prs)

def draw_loss(train_ls: List[float], test_ls: List[float], run_dir: str):
    plt.figure(figsize=(14, 8))
    plt.plot(train_ls, label='train')
    plt.plot(test_ls, label='test')
    plt.legend()
    save_show(os.path.join(run_dir, 'loss.png'), 600, show=False)
    plt.ylim(0, 3)
    save_show(os.path.join(run_dir, 'loss_zoom.png'), 600, show=False)
    
def train_one_fold(run_dir: str, fold: int, train_ds, test_ds, valid_ds, input_dim, args: GlobalSettings):
    # create model
    model = init_model_parameter(Attn3(input_dim, 3, 1, num_layers=1, nhead=8, dropout=0.4)).cuda()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    # train
    best_loss, best_paths_queue = 1e10, queue.Queue(maxsize=1)
    best_val_loss, best_val_paths_queue = 1e10, queue.Queue(maxsize=1)
    train_ls, test_ls = [], []
    for epoch in range(50000):
        losses = AverageMeter('Train', ':7.5f')
        test_losses = AverageMeter('Test', ':7.5f')
        valid_losses = AverageMeter('Valid', ':7.5f')
        best_losses = AverageMeter('Best', ':7.5f')
        train_pr = AverageMeter('Train-Pr', ':7.5f')
        test_pr = AverageMeter('Test-Pr', ':7.5f')
        valid_pr = AverageMeter('Valid-Pr', ':7.5f')
        best_val_losses = AverageMeter('Best-Val', ':7.5f')
        sum_batchs = len(train_ds)
        progress = ProgressMeter(sum_batchs, [losses, test_losses, best_losses, train_pr,
                                              test_pr, valid_losses, valid_pr],
                                 prefix="Epoch: [{}]".format(epoch), mp = args.mp)
        # train
        for i, (x, y) in enumerate(train_ds):
            loss, train_pr_ = train_batch(model, optimizer, criterion, x, y)
            train_pr.update(train_pr_, 1)
            losses.update(loss, 1)
        train_ls.append(losses.avg)
        # test
        for i, (x, y) in enumerate(test_ds):
            loss, test_pr_ = test_batch(model, criterion, x, y)
            test_pr.update(test_pr_, 1)
            test_losses.update(loss, 1)
        test_ls.append(test_losses.avg)
        # valid
        for i, (x, y) in enumerate(valid_ds):
            loss, valid_pr_ = test_batch(model, criterion, x, y)
            valid_pr.update(valid_pr_, 1)
            valid_losses.update(loss, 1)
        # save model if test loss is best
        best_losses.val = best_loss
        if test_losses.avg < best_loss:
            best_losses.val = best_loss = test_losses.avg
            path = save_checkpoint(epoch, args, model, optimizer, best_loss,
                                   {'train_pr': train_pr.avg, 'test_pr': test_pr.avg}, tailName=f'K{fold}_{best_loss}_best')
            if best_paths_queue.full():
                os.remove(best_paths_queue.get())
            best_paths_queue.put(path)
            args.mp.mprint(f'best model saved at test loss {best_loss:.5f}')
        # save model if valid loss is best
        if valid_losses.avg < best_val_loss:
            best_val_losses.val = best_val_loss = valid_losses.avg
            path = save_checkpoint(epoch, args, model, optimizer, best_val_loss,
                                   {'train_pr': train_pr.avg, 'test_pr': test_pr.avg}, tailName=f'K{fold}_{best_val_loss}_best_val')
            if best_val_paths_queue.full():
                os.remove(best_val_paths_queue.get())
            best_val_paths_queue.put(path)
            args.mp.mprint(f'best model saved at valid loss {best_val_loss:.5f}')
        progress.display(i)
    # save model
    draw_loss(train_ls, test_ls, run_dir)
    save_checkpoint(epoch, args, model, optimizer, loss, {'train_loss': train_ls, 'test_loss': test_ls}, tailName=f'K{fold}_final')
    args.mp.exit()
    # return stats
    return best_loss, best_val_loss

def get_dataset(data, labels: torch.Tensor, args: GlobalSettings, idx: np.array):
    x = [(h_i, m_i) for h_i, m_i in zip(data['hidden_states'][idx].cuda(), data['attention_mask'][idx].cuda())]
    return DataSetRAM(args, device='cuda', x=x, y=labels[idx]).split([0, 1], drop_last = False)[0]

def train(k_fold: int = 5):
    root = str(ROOT / 'MuMoPepcan/runs')
    run_dir = os.path.join(root, get_fmt_time())
    os.makedirs(run_dir)
    shutil.copy(__file__, run_dir)
    # load data
    ## PepDoRA: [50, 343, 384], ChemBERT: [50, 374, 768]
    data = torch.load(str(ROOT / 'MuMoPepcan/data/processed_data/ligand_feats_PepDoRA_seq.pt'))
    labels = pd.read_excel(str(ROOT / 'MuMoPepcan/data/label/wet_exp.xlsx'), index_col='name')
    labels = torch.tensor(labels.loc[list(data['name']),
                                     ['animal', 'cell_100', 'cell_300', 'cell_1000']].values,
                          dtype=torch.float32).cuda()
    idx_train = np.arange(len(data['name']))[np.array(list(map(lambda x: not x.startswith('3'), data['name'])))]
    idx_valid = np.arange(len(data['name']))[np.array(list(map(lambda x: x.startswith('3'), data['name'])))]
    # run KFold
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=3407)
    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(idx_train)):
        args = GlobalSettings(Mprint(os.path.join(run_dir, f'KFold_{fold}_log.txt')), run_dir, seed=3407)
        args.batch_size = 16
        args.mp.top_string = f'KFold {fold}'
        train_ds = get_dataset(data, labels, args, idx_train[train_idx])
        test_ds = get_dataset(data, labels, args, idx_train[test_idx])
        valid_ds = get_dataset(data, labels, args, idx_valid)
        # train
        k_loss, k_val_loss = train_one_fold(run_dir, fold, train_ds, test_ds, valid_ds, data['hidden_states'].shape[2], args)
        fold_results.append((k_loss, k_val_loss))
    df = pd.DataFrame(fold_results, columns=['test_loss', 'valid_loss'])
    df.to_excel(os.path.join(run_dir, 'KFold_results.xlsx'), index=False)
    
@torch.no_grad()
def vis_attn(run_dir: str, sub_name: str = 'best'):
    # load model
    paths = get_paths_with_extension(run_dir, ['.pth.tar'], True, sub_name)
    args = GlobalSettings(Mprint(clean_first=False), run_dir, seed=3407)
    if len(paths) == 1:
        args.resume = paths[0]
    else:
        args.resume = paths[np.argmin([float(os.path.basename(path).split('_')[1]) for path in paths])]
    model = Attn(32, 3, 1, 0.3).cuda()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    resume_checkpoint(args, model, optimizer)
    model.eval()
    # load data
    df = pd.read_excel('/home/pcmd36/Desktop/BHM/Others/nn_dock/data/res_freq_line_df.xlsx')
    attn_df = pd.DataFrame(columns=['name'] + list(df.columns[6:]))
    for name in df['name'].unique():
        data = np.array(list(df.loc[df['name'] == name, list(df.columns[5:])].values))
        x = torch.tensor(data, dtype=torch.float32).cuda()
        x[:, 1:] = torch.where(x[:, 1:] > 1, 1, x[:, 1:])
        x *= torch.arange(1, x.shape[1]+1, device='cuda').reshape(1, -1)
        energy = x[:, 0].reshape(-1, 1, 1).to(torch.float32)
        res = x[:, 1:].to(torch.int64)
        out, attn = model(energy, res)
        attn = attn[1].cpu().detach().numpy().mean(axis=0).mean(axis=0).reshape(-1)
        # attn = attn / attn.sum() * 100
        attn_df.loc[len(attn_df), :] = [name] + list(attn)
    attn_df.to_excel(os.path.join(run_dir, f'attn1_{sub_name}.xlsx'), index=False)
    attn_df.set_index('name', drop=True, inplace=True)
    attn_df = attn_df.astype(float)
    # vitualize
    fig = plt.figure(figsize=(10, 10))
    sns.heatmap(attn_df, xticklabels=list(df.columns[6:]), yticklabels=df['name'].unique(), cmap='coolwarm', ax=fig.gca())
    save_show(os.path.join(run_dir, f'attn1_{sub_name}.png'), 600, show=False)
    

@torch.no_grad()
def evaluate_kfold_model(run_dir: str, sub_name: str = 'best'):
    run_dir = get_dir(ROOT / 'MuMoPepcan/runs', 1, dir_name_substr=run_dir)[0]
    # load data
    data = torch.load(str(ROOT / 'MuMoPepcan/data/processed_data/ligand_feats_PepDoRA_seq.pt'))
    labels = pd.read_excel(str(ROOT / 'MuMoPepcan/data/label/wet_exp.xlsx'), index_col='name')
    labels = torch.tensor(labels.loc[list(data['name']),
                                 ['animal', 'cell_100', 'cell_300', 'cell_1000']].values,
                      dtype=torch.float32).cuda()
    
    # prepare index
    idx_train = np.arange(len(data['name']))[np.array(list(map(lambda x: not x.startswith('3'), data['name'])))]
    idx_valid = np.arange(len(data['name']))[np.array(list(map(lambda x: x.startswith('3'), data['name'])))]
    
    # init result container
    test_losses, valid_losses = [], []
    test_prs, valid_prs = [], []
    criterion = nn.MSELoss()
    
    # eval each fold
    kf = KFold(n_splits=5, shuffle=True, random_state=3407)
    for fold, (train_idx, test_idx) in enumerate(kf.split(idx_train)):
        # get model path
        ckp_path = glob(os.path.join(run_dir, f'checkpoint_K{fold}_*_{sub_name}_2025*'))[0]
        # load model
        args = GlobalSettings(Mprint(clean_first=False), run_dir, seed=3407)
        args.batch_size = 16
        args.resume = ckp_path
        model = Attn(data['hidden_states'].shape[2], 3, 1, num_layers=1, nhead=8, dropout=0.4).cuda()
        optimizer = AdamW(model.parameters(), lr=1e-3)
        resume_checkpoint(args, model, optimizer)
        model.eval()
        
        # get dataset
        test_ds = get_dataset(data, labels, args, idx_train[test_idx])
        valid_ds = get_dataset(data, labels, args, idx_valid)
        
        # eval on test set
        test_fold_losses, test_fold_prs = [], []
        for x, y in test_ds:
            loss, pr = test_batch(model, criterion, x, y)
            test_fold_losses.append(loss)
            test_fold_prs.append(pr)
        test_losses.append(np.mean(test_fold_losses))
        test_prs.append(np.mean(test_fold_prs))
        
        # eval on valid set
        valid_fold_losses, valid_fold_prs = [], []
        for x, y in valid_ds:
            loss, pr = test_batch(model, criterion, x, y)
            valid_fold_losses.append(loss)
            valid_fold_prs.append(pr)
        valid_losses.append(np.mean(valid_fold_losses))
        valid_prs.append(np.mean(valid_fold_prs))
    
    # calculate mean and std
    results = {
        'test_loss_mean': np.mean(test_losses),
        'test_loss_std': np.std(test_losses),
        'test_pr_mean': np.mean(test_prs),
        'test_pr_std': np.std(test_prs),
        'valid_loss_mean': np.mean(valid_losses),
        'valid_loss_std': np.std(valid_losses),
        'valid_pr_mean': np.mean(valid_prs),
        'valid_pr_std': np.std(valid_prs)
    }
    
    # 保存结果到Excel
    df = pd.DataFrame([results])
    df.to_excel(os.path.join(run_dir, f'evaluation_results_{sub_name}.xlsx'), index=False)
    
    return results


if __name__ == '__main__':
    train(k_fold=5)
    # evaluate_kfold_model('2025-05-23 08-08-25.423160', 'best_val')