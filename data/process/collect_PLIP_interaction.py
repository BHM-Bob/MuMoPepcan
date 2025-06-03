'''
Date: 2025-04-07 20:04:06
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-06-03 15:34:37
Description: 
'''
import platform
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from lazydock.pml.plip_interaction import SUPPORTED_MODE
from tqdm import tqdm

SERVER = platform.uname().node
ROOT = Path(f'/home/{SERVER}/Desktop/BHM/CB1-Pepcans-MDS/')

import sys

sys.path.append(str(ROOT / 'MuMoPepcan'))


def make_one_hot_df(data_df: pd.DataFrame) -> pd.DataFrame:
    data = defaultdict(lambda: defaultdict(float)) # {(series, name, ty): {res: value}}
    for index, row in tqdm(data_df.iterrows(), total=len(data_df)):
        series, name, time, t = row['series'], row['name'], int(row['time']), str(row['t'])
        key = (series, name, t, time)
        for ty in SUPPORTED_MODE:
            
            if pd.isna(row.get(ty, None)):
                continue
            
            entries = row[ty].split(';')
            for entry in entries:
                res_val = entry.strip().split('-')
                if len(res_val) != 2:
                    continue
                res, val = res_val
                resi = int(res[3:])
                if resi > 400:
                    continue # ignore H8 and C-Term
                if res in data[key]:
                    continue
                try:
                    data[key][res] = 1
                except ValueError:
                    continue
        # if line is empty, add a dummy entry
        if 'LEU111' not in data[key]:
            data[key]['LEU111'] = 0

    
    mat_df = pd.DataFrame.from_dict({k: dict(v) for k, v in data.items()}, orient='index')
    mat_df.index = pd.MultiIndex.from_tuples(mat_df.index, names=['series', 'name', 't', 'time'])
    mat_df = mat_df.fillna(0)
    mat_df.sort_index(axis=1, inplace=True, key=lambda x: x.str[3:].astype(int))
    mat_df.sort_index(axis=0, inplace=True)
    
    mat_data = {}
    names = mat_df.index.get_level_values('name').unique()
    for name in tqdm(names, desc='Processing names'):
        mat_data[name] = {}
        ts = mat_df.loc[pd.IndexSlice[:, name, :, :], :].index.get_level_values('t').unique()
        for t in tqdm(ts, desc=f'Processing {name}', leave=False):
            sub_df = mat_df.loc[pd.IndexSlice[:, name, t, :], :]
            mat_data[name][t] = torch.tensor(sub_df.values, dtype=torch.bool)
    mat_data['columns'] = list(mat_df.columns)
    torch.save(mat_data, ROOT/'MuMoPepcan/data/processed_data/MDS_plip_interactions_one_hot.pt')
    return mat_df


if __name__ == '__main__':
    data_df = pd.read_csv(str(ROOT / 'MuMoPepcan/data/processed_data/MDS_plip_interactions.csv'))
    mat_df = make_one_hot_df(data_df)