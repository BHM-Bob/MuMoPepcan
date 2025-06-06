
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
import torch
import torch.nn as nn
from mbapy.base import get_fmt_time
from mbapy.dl_torch.bb import RoPE
from mbapy.dl_torch.m import COneD, COneDLayer, LayerCfg, TransCfg
from mbapy.dl_torch.utils import (AverageMeter, GlobalSettings, Mprint,
                                  ProgressMeter, init_model_parameter,
                                  resume_checkpoint, save_checkpoint)
from mbapy.file import get_paths_with_extension, get_dir
from mbapy.plot import save_show
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

torch.set_float32_matmul_precision('high')

SERVER = platform.uname().node
ROOT = Path(f'/home/{SERVER}/Desktop/BHM/CB1-Pepcans-MDS/')

import sys

sys.path.append(str(ROOT / 'MuMoPepcan'))

from data.get_data import (MultiDataset, load_PLIP_one_hot, load_pose_data,
                           load_SMILES_feat, load_wet_exp_label)
from task.utils import draw_loss, get_parameter_number
from model.loss import BalancedFocalLoss
from model.MultiModel2 import arch1, arch2, arch3, arch4, arch5, arch6, arch7, arch8


def train_batch(model: nn.Module, optimizer, criterion1: nn.MSELoss, criterion2: nn.BCEWithLogitsLoss,
                pose: torch.Tensor, smiles: torch.Tensor, mask: torch.Tensor, plip: torch.Tensor, wet: torch.Tensor):
    # add noise
    smiles += 0.01*torch.randn_like(smiles, device=smiles.device) # [N, ]
    # add random mask
    mask = torch.where((mask != 0) & torch.rand(mask.shape, device=mask.device).gt(0.9), 0, mask)
    # train
    model.train()
    optimizer.zero_grad()
    wet_p, plip_p = model(pose, smiles, mask)
    # like label-smoothing, just add noise
    wet[:, 0] += 0.1*torch.randn_like(wet[:, 0], device=wet.device)
    wet[:, 1:] += 0.01*torch.randn_like(wet[:, 1:], device=wet.device)
    # scale label and out for cell exp
    wet[:, 1:] *= 20
    wet_p[:, 1:] *= 20
    # calculate loss and backpropagation
    loss1 = criterion1(wet_p, wet)
    loss2 = criterion2(plip_p, plip.to(dtype=torch.float32))
    f1 = f1_score(plip.cpu().numpy(), plip_p.sigmoid().gt(0.5).cpu().numpy(), average='weighted', zero_division=0)
    try:
        with torch.no_grad():
            scale = max(int(loss1.item()) // (int(loss2.item()) + 1), 1)
    except:
        scale = 1
    loss = loss1 + scale * loss2
    loss.backward()
    optimizer.step()
    return loss1.item(), loss2.item(), f1

def test_batch(model: nn.Module, criterion1: nn.MSELoss, criterion2: nn.BCEWithLogitsLoss,
                pose: torch.Tensor, smiles: torch.Tensor, mask: torch.Tensor, plip: torch.Tensor, wet: torch.Tensor):
    model.eval()
    with torch.no_grad():
        wet_p, plip_p = model(pose, smiles, mask)
        loss = criterion1(wet_p, wet)
        loss2 = criterion2(plip_p, plip.to(dtype=torch.float32))
        f1 = f1_score(plip.cpu().numpy(), plip_p.sigmoid().gt(0.5).cpu().numpy(), average='weighted', zero_division=0)
    return loss.item(), loss2.item(), f1
    
def train_one_fold(run_dir: str, fold: int, train_ds, test_ds, valid_ds, args: GlobalSettings):
    # create model
    hidden_dim, dropout = 384, 0.4
    # cnn_cfg = [
    #         LayerCfg( 3,   8,  8, 1, 'SABlock', avg_size=2, use_trans=False), # 32 ->16
    #         LayerCfg( 8,  16,  4, 1, 'SABlock', avg_size=2, use_trans=False), # 32 ->8
    #         LayerCfg(16,  32,  4, 1, 'SABlock', avg_size=1, use_trans=True, # 8
    #                 trans_layer='EncoderLayer', trans_cfg=TransCfg(32, n_layers=1, dropout=dropout)),
    #         LayerCfg(32, hidden_dim,  4, 1, 'SABlock', avg_size=1, use_trans=True, # 8
    #                 trans_layer='EncoderLayer', trans_cfg=TransCfg(384, n_layers=1, dropout=dropout)),
    #         ]
    # model = arch7.MultiModel(args, cnn_cfg, 1024-932, hidden_dim, 4, train_ds.dataset.plip.shape[1], n_layer=1,
    #                          n_head=8, dropout=dropout, RoPE = [False, False, False, True])
    
    cnn_cfg = [
            LayerCfg( 3,   4, 32, 1, 'SABlock1D', avg_size=1, use_trans=False), # 1212 -> 1212
            LayerCfg( 4,   8, 16, 1, 'SABlock1D', avg_size=4, use_trans=False), # 1212 -> 303
            LayerCfg( 8,  16, 16, 1, 'SABlock1D', avg_size=1, use_trans=True, # 303 -> 303
                    trans_layer='EncoderLayer', trans_cfg=TransCfg(16, n_layers=1, dropout=dropout)),
            LayerCfg(16, hidden_dim,  16, 1, 'SABlock1D', avg_size=1, use_trans=True, # 303 -> 303
                    trans_layer='EncoderLayer', trans_cfg=TransCfg(384, n_layers=1, dropout=dropout)),
            ]
    model = arch8.MultiModel(args, cnn_cfg, hidden_dim, 4, train_ds.dataset.plip.shape[1], n_layer=1,
                             n_head=8, dropout=dropout, RoPE = [False, False, False, True])
    model = init_model_parameter(model).cuda()
    args.mp.mprint(str(model))
    model = torch.compile(model)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion1 = nn.MSELoss()
    # calculate pos_weight for criterion2
    num_pos = train_ds.dataset.plip.sum(dim=0)  # [L]
    num_neg = train_ds.dataset.plip.shape[0] - num_pos
    criterion2 = BalancedFocalLoss(pos_weight=(num_neg / num_pos.clamp(min=1)).cuda())
    # train
    best_loss, best_paths_queue = 1e10, queue.Queue(maxsize=1)
    best_val_loss, best_val_paths_queue = 1e10, queue.Queue(maxsize=1)
    train_ls, test_ls, valid_ls = [], [], []
    for epoch in range(20):
        losses = AverageMeter('Train', ':7.5f')
        losses2 = AverageMeter('Train2', ':7.5f')
        train_f1 = AverageMeter('F1', ':7.5f')
        test_losses = AverageMeter('Test1', ':7.5f')
        test2_losses = AverageMeter('Test2', ':7.5f')
        test2_f1 = AverageMeter('F1', ':7.5f')
        valid_losses = AverageMeter('Valid', ':7.5f')
        valid2_losses = AverageMeter('Valid2', ':7.5f')
        valid2_f1 = AverageMeter('F1', ':7.5f')
        best_losses = AverageMeter('Best', ':7.5f')
        best_val_losses = AverageMeter('Best-Val', ':7.5f')
        train_progress = ProgressMeter(len(train_ds), [losses, losses2, train_f1, best_losses, best_val_losses],
                                       prefix="Train: [{}]".format(epoch), mp = args.mp)
        test_progress = ProgressMeter(len(test_ds), [test_losses, test2_losses, test2_f1, best_losses],
                                      prefix="Test: [{}]".format(epoch), mp = args.mp)
        valid_progress = ProgressMeter(len(valid_ds), [valid_losses, valid2_losses, valid2_f1, best_val_losses],
                                       prefix="Valid: [{}]".format(epoch), mp = args.mp)
        best_losses.val = best_loss
        best_val_losses.val = best_val_loss
        # train
        for i, (pose, smiles, mask, plip, wet) in enumerate(train_ds):
            loss, loss2, f1 = train_batch(model, optimizer, criterion1, criterion2,
                                            pose.cuda(), smiles.cuda(), mask.cuda(), plip.cuda(), wet.cuda())
            losses.update(loss, 1)
            losses2.update(loss2, 1)
            train_f1.update(f1, 1)
            if i % 10 == 0:
                train_progress.display(i)
        train_ls.append(losses.avg)
        # test
        for i, (pose, smiles, mask, plip, wet) in enumerate(test_ds):
            loss, loss2, f1 = test_batch(model, criterion1, criterion2,
                                     pose.cuda(), smiles.cuda(), mask.cuda(), plip.cuda(), wet.cuda())
            test_losses.update(loss, 1)
            test2_losses.update(loss2, 1)
            test2_f1.update(f1, 1)
            if i % 10 == 0:
                test_progress.display(i)
        test_ls.append(test_losses.avg)
        # valid
        for i, (pose, smiles, mask, plip, wet) in enumerate(valid_ds):
            loss, loss2, f1 = test_batch(model, criterion1, criterion2,
                              pose.cuda(), smiles.cuda(), mask.cuda(), plip.cuda(), wet.cuda())
            valid_losses.update(loss, 1)
            valid2_losses.update(loss2, 1)
            valid2_f1.update(f1, 1)
            if i % 10 == 0:
                valid_progress.display(i)
        valid_ls.append(valid_losses.avg)
        # save model if test loss is best
        best_losses.val = best_loss
        if test_losses.avg < best_loss:
            best_losses.val = best_loss = test_losses.avg
            path = save_checkpoint(epoch, args, model, optimizer, best_loss,
                                   {'train_test': train_ls, 'test': test_ls}, tailName=f'K{fold}_{best_loss}_best')
            if best_paths_queue.full():
                os.remove(best_paths_queue.get())
            best_paths_queue.put(path)
            args.mp.mprint(f'best model saved at test loss {best_loss:.5f}')
        # save model if valid loss is best
        if valid_losses.avg < best_val_loss:
            best_val_losses.val = best_val_loss = valid_losses.avg
            path = save_checkpoint(epoch, args, model, optimizer, best_val_loss,
                                   {'train_test': train_ls, 'test': test_ls}, tailName=f'K{fold}_{best_val_loss}_best_val')
            if best_val_paths_queue.full():
                os.remove(best_val_paths_queue.get())
            best_val_paths_queue.put(path)
            args.mp.mprint(f'best model saved at valid loss {best_val_loss:.5f}')
    # save model
    draw_loss(train_ls, test_ls, valid_ls, run_dir, fold)
    save_checkpoint(epoch, args, model, optimizer, loss, {'train_loss': train_ls, 'test_loss': test_ls}, tailName=f'K{fold}_final')
    args.mp.exit()
    # return stats
    return best_loss, best_val_loss, train_ls[-1], len(train_ls)

def get_dataset(pose_train_data, smiles_data, smiles_mask, smile_train_idx, plip_train_data, wet_label, wet_train_idx, train_idx, args: GlobalSettings):
    if train_idx is None:
        train_idx = np.arange(len(pose_train_data))
    dataset = MultiDataset(pose_train_data[train_idx], smiles_data.clone(), smiles_mask.clone(), smile_train_idx[train_idx],
                           plip_train_data[train_idx], wet_label, wet_train_idx[train_idx])
    args.mp.mprint(f'dataset: {len(dataset)}')
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)

def train(k_fold: int = 5):
    root = str(ROOT / 'MuMoPepcan/runs')
    run_dir = os.path.join(root, get_fmt_time())
    os.makedirs(run_dir)
    shutil.copy(__file__, run_dir)
    shutil.copytree(Path(__file__).parent / '../../model', Path(run_dir) / 'model')
    # load data
    ## PepDoRA: [50, 343, 384], ChemBERT: [50, 374, 768]
    traj_st = 5000
    idxs = np.arange(138 * traj_st)
    # run KFold
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=3407)
    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(idxs)):
        # global settings
        args = GlobalSettings(Mprint(os.path.join(run_dir, f'KFold_{fold}_log.txt')), run_dir, seed=3407)
        args.batch_size = 256
        args.mp.top_string = f'KFold {fold}'
        # load data
        pose_train_data, pose_valid_data = load_pose_data(frame_start=traj_st+1, frame_end=10001, frame_step=1)
        plip_train_data, plip_valid_data = load_PLIP_one_hot(frame_start=traj_st, frame_end=10000, frame_step=1)
        smiles_data, smiles_mask, smile_train_idx, smile_valid_idx = load_SMILES_feat('PepDoRA_seq', frame_start=traj_st, frame_end=10000, frame_step=1)
        wet_label, wet_train_idx, wet_valid_idx = load_wet_exp_label(frame_start=traj_st, frame_end=10000, frame_step=1)
        train_ds = get_dataset(pose_train_data, smiles_data, smiles_mask, smile_train_idx,
                               plip_train_data, wet_label, wet_train_idx, train_idx, args)
        test_ds = get_dataset(pose_train_data, smiles_data, smiles_mask, smile_train_idx,
                              plip_train_data, wet_label, wet_train_idx, test_idx, args)
        valid_ds = get_dataset(pose_valid_data, smiles_data, smiles_mask, smile_valid_idx,
                               plip_valid_data, wet_label, wet_valid_idx, None, args)
        # train
        k_loss, k_val_loss, k_train_loss, k_n_epoch = train_one_fold(run_dir, fold, train_ds, test_ds, valid_ds, args)
        fold_results.append((k_loss, k_val_loss, k_train_loss, k_n_epoch))
        # clean up
        del pose_train_data, pose_valid_data, plip_train_data, plip_valid_data
        del smiles_data, smiles_mask, smile_train_idx, smile_valid_idx
        del wet_label, wet_train_idx, wet_valid_idx
        del train_ds, test_ds, valid_ds
        torch.cuda.empty_cache()
        # break to train only one-fold
        # break
    df = pd.DataFrame(fold_results, columns=['test_loss', 'valid_loss', 'final train', 'n epoch'])
    df.to_excel(os.path.join(run_dir, 'KFold_results.xlsx'), index=False)
    

@torch.no_grad()
def evaluate_kfold_model(run_dir: str, sub_name: str = 'best'):
    run_dir = get_dir(ROOT / 'MuMoPepcan/runs', 1, dir_name_substr=run_dir)[0]
    # init result container
    test_losses, valid_losses = [], []
    test_prs, valid_prs = [], []
    
    # eval on each fold
    traj_st, traj_ed = 0, 5000
    idxs = np.arange(138 * (traj_ed - traj_st))
    kf = KFold(n_splits=5, shuffle=True, random_state=3407)
    for fold, (train_idx, test_idx) in enumerate(kf.split(idxs)):
        # init args
        args = GlobalSettings(Mprint(clean_first=False), run_dir, seed=3407)
        args.batch_size = 256
        args.resume = glob(os.path.join(run_dir, f'checkpoint_K{fold}_*_{sub_name}_2025*'))[0]
        # load data
        pose_train_data, pose_valid_data = load_pose_data(frame_start=traj_st+1, frame_end=traj_ed+1, frame_step=1)
        plip_train_data, plip_valid_data = load_PLIP_one_hot(frame_start=traj_st, frame_end=traj_ed, frame_step=1)
        smiles_data, smiles_mask, smile_train_idx, smile_valid_idx = load_SMILES_feat('PepDoRA_seq', frame_start=traj_st, frame_end=traj_ed, frame_step=1)
        wet_label, wet_train_idx, wet_valid_idx = load_wet_exp_label(frame_start=traj_st, frame_end=traj_ed, frame_step=1)
        train_ds = get_dataset(pose_train_data, smiles_data, smiles_mask, smile_train_idx,
                               plip_train_data, wet_label, wet_train_idx, train_idx, args)
        test_ds = get_dataset(pose_train_data, smiles_data, smiles_mask, smile_train_idx,
                                plip_train_data, wet_label, wet_train_idx, test_idx, args)
        valid_ds = get_dataset(pose_valid_data, smiles_data, smiles_mask, smile_valid_idx,
                                plip_valid_data, wet_label, wet_valid_idx, None, args)
        # create model
        hidden_dim, dropout = 384, 0.4
        cnn_cfg = [
                LayerCfg(  3,  16,  8, 1, 'SABlock', avg_size=2, use_trans=False), # 32 ->16
                LayerCfg( 16,  32,  4, 1, 'SABlock', avg_size=2, use_trans=False), # 32 ->8
                LayerCfg( 32, 128,  4, 1, 'SABlock', avg_size=1, use_trans=True, # 8
                        trans_layer='EncoderLayer', trans_cfg=TransCfg(128, n_layers=1, dropout=dropout)),
                LayerCfg(128, hidden_dim,  4, 1, 'SABlock', avg_size=1, use_trans=True, # 8
                        trans_layer='EncoderLayer', trans_cfg=TransCfg(hidden_dim, n_layers=1, dropout=dropout)),
                ]
        model = arch2.MultiModel(args, cnn_cfg, 1024-932, hidden_dim, 4, train_ds.dataset.plip.shape[1], n_layer=1,
                                n_head=8, dropout=dropout, RoPE = [False, False, False, True]).cuda()
        
        # cnn_cfg = [
        #         LayerCfg( 3,   8, 32, 1, 'SABlock1D', avg_size=1, use_trans=False), # 1212 -> 1212
        #         LayerCfg( 8,  16, 16, 1, 'SABlock1D', avg_size=4, use_trans=False), # 1212 -> 303
        #         LayerCfg(16,  32, 16, 1, 'SABlock1D', avg_size=1, use_trans=True, # 303 -> 303
        #                 trans_layer='EncoderLayer', trans_cfg=TransCfg(32, n_layers=1, dropout=dropout)),
        #         LayerCfg(32, hidden_dim,  16, 1, 'SABlock1D', avg_size=1, use_trans=True, # 303 -> 303
        #                 trans_layer='EncoderLayer', trans_cfg=TransCfg(384, n_layers=1, dropout=dropout)),
        #         ]
        # model = arch1.MultiModel(args, cnn_cfg, hidden_dim, 4, train_ds.dataset.plip.shape[1], n_layer=1,
        #                         n_head=8, dropout=dropout, RoPE = [False, False, False, True]).cuda()
        print(get_parameter_number(model))
        model = torch.compile(model)
        resume_checkpoint(args, model, None, False)
        model.eval()
        
        # loss function
        criterion1 = nn.MSELoss()
        num_pos = train_ds.dataset.plip.sum(dim=0)  # [L]
        num_neg = train_ds.dataset.plip.shape[0] - num_pos
        criterion2 = BalancedFocalLoss(pos_weight=(num_neg / num_pos.clamp(min=1)).cuda())
        
        # evaluate on test set
        test_fold_losses, test_fold_prs = [], []
        for pose, smiles, mask, plip, wet in tqdm(test_ds, total=len(test_ds), desc='test'):
            loss, loss2, f1, wet_Rp = test_batch(model, criterion1, criterion2,
                                     pose.cuda(), smiles.cuda(), mask.cuda(), plip.cuda(), wet.cuda())
            test_fold_losses.append(loss)
            test_fold_prs.append(wet_Rp)
        test_losses.append(np.mean(test_fold_losses))
        test_prs.append(np.mean(test_fold_prs))
        
        # eval on valid set
        valid_fold_losses, valid_fold_prs = [], []
        for pose, smiles, mask, plip, wet in tqdm(valid_ds, total=len(valid_ds), desc='valid'):
            loss, loss2, f1, wet_Rp = test_batch(model, criterion1, criterion2,
                                     pose.cuda(), smiles.cuda(), mask.cuda(), plip.cuda(), wet.cuda())
            valid_fold_losses.append(loss)
            valid_fold_prs.append(wet_Rp)
        valid_losses.append(np.mean(valid_fold_losses))
        valid_prs.append(np.mean(valid_fold_prs))
        
        print(f'fold {fold} test loss: {np.mean(test_fold_losses):.4f}, test pr: {np.mean(test_fold_prs):.4f}')
        print(f'fold {fold} valid loss: {np.mean(valid_fold_losses):.4f}, valid pr: {np.mean(valid_fold_prs):.4f}')
    
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
    
    df = pd.DataFrame([results])
    df.to_excel(os.path.join(run_dir, f'evaluation_results_{sub_name}.xlsx'), index=False)
    
    return results


@torch.no_grad()
def viz_attn(run_dir: str, sub_name: str = 'best'):
    run_dir = get_dir(ROOT / 'MuMoPepcan/runs', 1, dir_name_substr=run_dir)[0]
    # eval on each fold
    traj_st, traj_ed = 5000, 10000
    idxs = np.arange(138 * (traj_ed - traj_st))
    # init args
    args = GlobalSettings(Mprint(clean_first=False), run_dir, seed=3407)
    args.batch_size = 300
    all_attn = {i:[] for i in range(50)}
    for fold in range(5):
        args.resume = glob(os.path.join(run_dir, f'checkpoint_K{fold}_*_{sub_name}_2025*'))[0]
        # load data
        pose_train_data, pose_valid_data = load_pose_data(frame_start=traj_st+1, frame_end=traj_ed+1, frame_step=1)
        plip_train_data, plip_valid_data = load_PLIP_one_hot(frame_start=traj_st, frame_end=traj_ed, frame_step=1)
        smiles_data, smiles_mask, smile_train_idx, smile_valid_idx = load_SMILES_feat('PepDoRA_seq', frame_start=traj_st, frame_end=traj_ed, frame_step=1)
        wet_label, wet_train_idx, wet_valid_idx = load_wet_exp_label(frame_start=traj_st, frame_end=traj_ed, frame_step=1)
        train_ds = get_dataset(pose_train_data, smiles_data, smiles_mask, smile_train_idx,
                                plip_train_data, wet_label, wet_train_idx, idxs, args, False, False)
        valid_ds = get_dataset(pose_valid_data, smiles_data, smiles_mask, smile_valid_idx,
                                plip_valid_data, wet_label, wet_valid_idx, None, args, False, False)
        # create model
        hidden_dim, dropout = 384, 0.4
        # cnn_cfg = [
        #         LayerCfg( 3,   8,  8, 1, 'SABlock', avg_size=2, use_trans=False), # 32 ->16
        #         LayerCfg( 8,  16,  4, 1, 'SABlock', avg_size=2, use_trans=False), # 32 ->8
        #         LayerCfg(16,  32,  4, 1, 'SABlock', avg_size=1, use_trans=True, # 8
        #                 trans_layer='EncoderLayer', trans_cfg=TransCfg(32, n_layers=1, dropout=dropout)),
        #         LayerCfg(32, hidden_dim,  4, 1, 'SABlock', avg_size=1, use_trans=True, # 8
        #                 trans_layer='EncoderLayer', trans_cfg=TransCfg(384, n_layers=1, dropout=dropout)),
        #         ]
        # model = arch7.MultiModel(args, cnn_cfg, 1024-932, hidden_dim, 4, train_ds.dataset.plip.shape[1], n_layer=1,
        #                          n_head=8, dropout=dropout, RoPE = [False, False, False, True])
        cnn_cfg = [
                LayerCfg( 3,   8, 32, 1, 'SABlock1D', avg_size=1, use_trans=False), # 1212 -> 1212
                LayerCfg( 8,  16, 16, 1, 'SABlock1D', avg_size=4, use_trans=False), # 1212 -> 303
                LayerCfg(16,  32, 16, 1, 'SABlock1D', avg_size=1, use_trans=False, # 303 -> 303
                        trans_layer='EncoderLayer', trans_cfg=TransCfg(32, n_layers=1, dropout=dropout)),
                LayerCfg(32, hidden_dim,  16, 1, 'SABlock1D', avg_size=1, use_trans=True, # 303 -> 303
                        trans_layer='EncoderLayer', trans_cfg=TransCfg(384, n_layers=1, dropout=dropout)),
                ]
        model = arch1.MultiModel(args, cnn_cfg, hidden_dim, 4, train_ds.dataset.plip.shape[1], n_layer=1,
                                n_head=8, dropout=dropout, RoPE = [False, False, False, True]).cuda()
        model = torch.compile(model)
        resume_checkpoint(args, model, None, False)
        model.eval()
        
        if sub_name == 'best':
            for i, (pose, smiles, mask, plip, wet) in tqdm(enumerate(train_ds), total=len(train_ds), desc='test'):
                attn = model.get_attn(pose.cuda(), smiles.cuda(), mask.cuda())
                pep_idx = int(i*args.batch_size/15000)
                all_attn[pep_idx].append(attn.cpu())
        elif sub_name == 'best_val':
            for i, (pose, smiles, mask, plip, wet) in tqdm(enumerate(valid_ds), total=len(valid_ds), desc='val'):
                attn = model.get_attn(pose.cuda(), smiles.cuda(), mask.cuda())
                pep_idx = int(i*args.batch_size/15000)+45
                all_attn[pep_idx].append(attn.cpu())
    for i in all_attn:
        if all_attn[i]:
            all_attn[i] = torch.cat(all_attn[i], dim=0).squeeze(1).mean(dim=0)
    torch.save(all_attn, os.path.join(run_dir, f'attn_{sub_name}.pt'))


if __name__ == '__main__':
    train(k_fold=5)
    # evaluate_kfold_model('2025-01-13 10-00-01.114011', 'best')
    # viz_attn('2025-01-13 10-00-01.114011', 'best')