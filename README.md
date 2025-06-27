<!--
 * @Date: 2025-06-03 11:01:00
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2025-06-26 22:34:11
 * @Description: 
-->
## MuMoPepcan: A Multilevel Multimodal Multitask Model for CB1R Targeted Peptide Biologic Activity Prediction

This is the official code repository for the paper: MuMoPepcan: A Multilevel Multimodal Multitask Model for CB1R Targeted Peptide Biologic Activity Prediction.

You can use this code to reproduce the MuMoPepcan results in the paper, AND you can even extend our work to your interested receptors!


### folder structure
```text
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
├── LICENSE: the license of this code
├── README.md: this file
├── requirements_DL.txt: the requirements for deep learning
├── requirements_MDS.txt: the requirements for MDS running and analysis
```

------

### Get start: 1. Reproduce the result in our paper
#### 1. download code files
```bash
mkdir CB1-Pepcans-MDS
cd CB1-Pepcans-MDS
git clone https://github.com/BHM-Bob/MuMoPepcan --depth=1
```

#### 2. download dataset
We provide processed coordinates and PLIP analysis results in [Zenodo](https://zenodo.org/records/15734130).
Asuming you are in `CB1-Pepcans-MDS` folder, you can download the data with:
```bash
mkdir MuMoPepcan/data/processed_data
wget -O MuMoPepcan/data/processed_data/aligned_pos.pt https://zenodo.org/record/15734130/files/aligned_pos.pt
wget -O MuMoPepcan/data/processed_data/aligned_TM_pos.pt https://zenodo.org/record/15734130/files/aligned_TM_pos.pt
wget -O MuMoPepcan/data/processed_data/ligand_feats_PepDoRA_seq.pt https://zenodo.org/record/15734130/files/ligand_feats_PepDoRA_seq.pt
wget -O MuMoPepcan/data/processed_data/MDS_plip_interactions_one_hot.pt https://zenodo.org/record/15734130/files/MDS_plip_interactions_one_hot.pt
```

#### 3. prepare python environment
You can modify the packages' version in `requirements_DL.txt` to fit your environment.
```
pip install -r requirements_DL.txt
```
BTW, if you already have a conda environment with `pytorch>2.0`, you can just skip this step.

#### 4. train MuMoPepcan
Asuming you are in `CB1-Pepcans-MDS` folder, you can use bellow commands:
- `python MuMoPepcan/task/train_mm1_traditional.py -h`
- `python MuMoPepcan/task/train_mm1.py -h`
- `python MuMoPepcan/task/train_mm2.py -h`

Forgive that there hasn't been a config system yet. You can comment (and uncomment) the code to define the model's architecture you want to train.

------

### Get start: 2. Extend our architecture to your interested peptides targeting CB1R
#### 1. download code files
As descripted in `Get start: 1. Reproduce the result in our paper`.

#### 2. download SMILES model
Download [ChemBERTa-77M-MLM](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM), [PepDoRA](https://huggingface.co/ChatterjeeLab/PepDoRA/tree/main), [PubChem10M_SMILES_BPE_450k](https://huggingface.co/seyonec/PubChem10M_SMILES_BPE_450k) from HuggingFace, and put them into `CB1-Pepcans-MDS/MuMoPepcan/pretrained` in each named folder.

#### 3. get a trained MuMoPepcan model
- Choice 1. train by yourself
As descripted in `Get start: 1. Reproduce the result in our paper`.
- Choice 2. download our trained model
We provide one best checkpoint in [Zenodo](https://zenodo.org/records/15734130).
Asuming you are in `CB1-Pepcans-MDS` folder, you can download the model with:
```bash
mkdir MuMoPepcan/runs
wget -O MuMoPepcan/runs/dual_modal_best_test.zip https://zenodo.org/record/15734130/files/dual_modal_best_test.zip
unzip MuMoPepcan/runs/dual_modal_best_test.zip -d MuMoPepcan/runs
```

#### 4. get SMILES feature by PepDoRA
The code for transform SMILES to feature is in `MuMoPepcan/data/process/collect_ligand_feat.py`.

------

### Get start: 3. Extend our architecture to your interested receptors
#### 1. download code files
As descripted in `Get start: 1. Reproduce the result in our paper`.

#### 2. download SMILES model
As descripted in `Get start: 2. Extend our architecture to your interested peptides targeting CB1R`.

#### 3. get SMILES feature by PepDoRA
As descripted in `Get start: 2. Extend our architecture to your interested peptides targeting CB1R`.

#### 4. prepare your own MDS dataset
1. Prepare your own MDS runs.
We describe the steps in the paper, you can take it as an example. Further more, our code accept a `.gro` file and a `.xtc` file as the trajectory of a ligand-receptor complex, which means you can run your own MDS in your way and get the aligned coordinates as MDS dataset.

Specially, we recommend you to use [LazyDock](https://github.com/BHM-Bob/LazyDock) to run your own MDS, while it is necessary to align the ligand and receptor and get PLIP analysis results in the same way as we do.

You can use follow command to install the requirements:
```bash
pip install -r requirements_MDS.txt
```

2. Get aligned coordinates and PLIP analysis results
- To align the ligand and receptor, you can use the code in `MuMoPepcan/data/process/collect_aligned_pos.py`.
- To get PLIP analysis results, you can use the command, specailly in the [docs](https://lazydock.readthedocs.io/en/latest/scripts/ana_gmx/):
```bash
lazydock-cli ana-gmx interaction -d PATH/TO/YOUR/FOLDER -top TPR_NAME -gro GRO_NAME -traj TRJ_NAME --receptor-chain-name Protein --ligand-chain-name LIG --alter-receptor-chain A --alter-ligand-chain Z --alter-ligand-res UNK --alter-ligand-atm HETATM --method plip --mode all --max-plot 24 -nw 8
```
*Where `PATH/TO/YOUR/FOLDER` is the path to your MDS folder, `TPR_NAME` is the name of your TPR file, `GRO_NAME` is the name of your GRO file, and `TRJ_NAME` is the name of your trajectory file. The other parameters are the same as we do in our paper.*
- To collect the PLIP analysis results, you can use the code in `MuMoPepcan/data/process/collect_plip_interactions.py`.
Note that our code ignore H8 and C-Term for CB1R by ignore the residue number larger than 400, you can also modify the code to fit your own dataset.


### Citation
`UNK`
