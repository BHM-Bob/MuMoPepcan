'''
Date: 2025-06-03 10:17:27
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-06-03 15:33:45
Description: 
'''
import platform
from pathlib import Path
from typing import Tuple

import pandas as pd
import rdkit
import torch
from mbapy.base import put_err
from peft import PeftConfig, PeftModel
from rdkit import Chem
from tqdm import tqdm
from transformers import (AutoModel, AutoModelForCausalLM,
                          AutoModelForMaskedLM, AutoTokenizer, RobertaModel,
                          RobertaTokenizer, pipeline)

SERVER = platform.uname().node
ROOT = Path(f'/home/{SERVER}/Desktop/BHM/CB1-Pepcans-MDS/')


def get_model(model_name: str):
    if model_name == 'ChemBERTa':
        model_path = str(ROOT / 'MuMoPepcan/pretrained/PubChem10M_SMILES_BPE_450k')
        model = AutoModelForMaskedLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif model_name == 'PepDoRA':
        base_model = str(ROOT / 'MuMoPepcan/pretrained/ChemBERTa-77M-MLM')
        adapter_model = str(ROOT / 'MuMoPepcan/pretrained/PepDoRA')
        model = AutoModelForCausalLM.from_pretrained(base_model)
        model = PeftModel.from_pretrained(model, adapter_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    else:
        raise ValueError(f'Unknown model name: {model_name}')
    return model, tokenizer


def get_SMILES(path: Path):
    mol = rdkit.Chem.rdmolfiles.MolFromPDBFile(str(path))
    return Chem.MolToSmiles(mol)


def get_SMILES_from_seq(seq: str):
    mol = rdkit.Chem.MolFromFASTA(seq)
    return Chem.MolToSmiles(mol)


def get_all_SMILES_from_pdb():
    path_df = pd.read_excel(ROOT / 'MuMoPepcan/data/paths.xlsx', sheet_name='final')
    feats = {}
    for name, path, t in tqdm(path_df.loc[:, ['name', 'path', 't']].values, total=len(path_df)):
        if name == 'single-receptor':
            continue
        if name not in {'WIN55212', 'AEA', '2-AG', 'AM251', 'AM6538'}:
            name = path.split('\\')[-1][:-2]
            suffix = 'aa'
        else:
            suffix = ''
        if name in feats:
            continue
        root = Path(ROOT / f'ligand/{name}{suffix}.pdb')
        SMILES = get_SMILES(root)
        if not SMILES:
            put_err(f'Failed to get SMILES for {root}')
            feats[name] = None
            continue
        feats[name] = SMILES
    return feats


def get_all_SMILES_from_seq():
    names, seq1 = pd.read_excel(ROOT / f'MuMoPepcan/data/ligs.xlsx').loc[:, ['name', 'seq1']].values.transpose(1, 0)
    smiles = list(map(lambda x: Chem.MolToSmiles(Chem.MolFromFASTA(x)), seq1))
    return {n[:-2]:s for n, s in zip(names, smiles)}
    

if __name__ == '__main__':
    method = 'PepDoRA'
    model, tokenizer = get_model(method)
    feats = get_all_SMILES_from_seq()
    feats_from_pdb = get_all_SMILES_from_pdb()
    for n in ['WIN55212', 'AEA', '2-AG', 'AM251', 'AM6538']:
        feats[n] = feats_from_pdb[n]
    # tokenize and get features
    inputs = tokenizer(list(feats.values()), return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # collect inputs and feature
    data = {'name': list(feats.keys()), 'SMILES': list(feats.values()),
            'logits': outputs.logits, 'hidden_states': outputs.hidden_states[-1],
            'inputs': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}
    torch.save(data, ROOT / f'MuMoPepcan/data/processed_data/ligand_feats_{method}_seq.pt')