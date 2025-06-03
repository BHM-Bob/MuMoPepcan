'''
Date: 2025-04-20 16:38:55
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-06-03 10:38:04
Description: 
'''
import os
import platform
from pathlib import Path

import pandas as pd
from mbapy.file import get_dir, get_paths_with_extension

SERVER = platform.uname().node
ROOT = Path(f'/home/{SERVER}/Desktop/BHM/CB1-Pepcans-MDS/')


model_root = str(ROOT / 'MuMoPepcan/runs')
results_path = str(ROOT / 'MuMoPepcan/runs/perf_stats.xlsx')

results_df = pd.DataFrame(columns=['folder', 'final_train', 'best_test', 'best_val', 'N_fold', 'server'])
results_df.set_index('folder', drop=True, inplace=True)
    
for md in get_dir(ROOT / 'MuMoPepcan/runs', 1, file_extensions=['xlsx'], item_name_substr='KFold_results.xlsx'):
    k_df = pd.read_excel(os.path.join(md, 'KFold_results.xlsx'))
    best_tests, best_valids = k_df['test_loss'].values, k_df['valid_loss'].values
    results_df.loc[os.path.basename(md), 'best_test'] = f'{best_tests.mean():6.4f} ({best_tests.std():6.4f})'
    results_df.loc[os.path.basename(md), 'best_val'] = f'{best_valids.mean():6.4f} ({best_valids.std():6.4f})'
    results_df.loc[os.path.basename(md), 'N_fold'] = best_tests.shape[0]
    if 'test_pr' in k_df.columns:
        test_prs = k_df['test_pr'].values
        results_df.loc[os.path.basename(md), 'test_pr'] = f'{test_prs.mean():6.4f} ({test_prs.std():6.4f})'
    if 'valid_pr' in k_df.columns:
        valid_prs = k_df['valid_pr'].values
        results_df.loc[os.path.basename(md), 'valid_pr'] = f'{valid_prs.mean():6.4f} ({valid_prs.std():6.4f})'
    server = get_paths_with_extension(md, ['py'], False, 'predictor_node')
    if server:
        server = os.path.basename(server[0]).replace('.py', '').replace('predictor_node', '')
    else:
        server = 'NA'
    if 'final trainAvg' in k_df.columns or 'final train' in k_df.columns:
        final_trains = k_df['final trainAvg'].values if 'final trainAvg' in k_df.columns else k_df['final train'].values
        results_df.loc[os.path.basename(md), 'final_train'] = f'{final_trains.mean():6.4f} ({final_trains.std():6.4f})'
    else:
        results_df.loc[os.path.basename(md), 'final_train'] = 'NA'
    results_df.loc[os.path.basename(md), 'server'] = server
        
results_df.sort_index(inplace=True)
results_df.to_excel(results_path)
print(results_df)
