<!--
 * @Date: 2025-06-03 11:01:00
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2025-06-03 15:18:41
 * @Description: 
-->
## MuMoPepcan: A Multilevel Multimodal Multitask Model for CB1R Targeted Peptide Biologic Activity Prediction

This is the official code repository for the paper: MuMoPepcan: A Multilevel Multimodal Multitask Model for CB1R Targeted Peptide Biologic Activity Prediction.

You can use this code to reproduce the MuMoPepcan results in the paper, AND you can even extend our work to your interested receptors!


### folder structure
```
MuMoPepcan
├── data
│   ├── label
│   │   ├── wet_exp.xlsx: the wet experiment data as the label
│   ├── process
│   │   ├── collect_aligned_pos.py: get aligned coordinates from MDS trajectory
│   │   ├── collect_ligand_feat.py: get ligand features from SMILES
│   │   ├── collect_PLIP_interaction.py: transfer PLIP interaction results to tensor
│   ├── processed_data
│   │   ├──...: processed data
│   ├── get_data.py: load each data
├── model
│   ├── ...: each network file
├── pretrained
│   ├── ...: pretrained SMILES model
├── task
│   ├── ...: train and evaluate each model

```

### Installation
#### 1. download code files
```bash
mkdir CB1-Pepcans-MDS
cd CB1-Pepcans-MDS
git clone https://github.com/BHM-Bob/MuMoPepcan
```
Forgive that we use absolute path to define the root directory in almost all files. You can replace `ROOT = Path(f'/home/{SERVER}/Desktop/BHM/CB1-Pepcans-MDS/')` to your own root directory in such VSCode tools.

#### 2. download SMILES model
Download `ChemBERTa-77M-MLM`, `PepDoRA`, `PubChem10M_SMILES_BPE_450k` from HuggingFace, and put them into `CB1-Pepcans-MDS/MuMoPepcan/pretrained` in each named folder.

#### 3. download MDS data
We provide processed coordinates and PLIP analysis results in `UNK`. After downloading, please put them into `CB1-Pepcans-MDS/data/processed_data`.


#### 4. prepare python environment
1. MDS data processing with [LazyDock](https://github.com/BHM-Bob/LazyDock)
```
pip install -r requirements_MDS.txt
```
2. train MuMoPepcan
You can modify the packages' version in `requirements_DL.txt` to fit your environment.
```
pip install -r requirements_DL.txt
```


### Usage
#### train MuMoPepcan

`python task/train_mm1_traditional.py`
`python task/train_mm1.py`
`python task/train_mm2.py`

Forgive that there hasn't been a argument parser yet. You can comment (and uncomment) the code to choose the model you want to train.


### Citation
`UNK`