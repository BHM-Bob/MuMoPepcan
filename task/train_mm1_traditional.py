import os
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mbapy.base import get_fmt_time
from mbapy.plot import save_show
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

SERVER = platform.uname().node
ROOT = Path(f'/home/{SERVER}/Desktop/BHM/CB1-Pepcans-MDS/')

def prepare_data():
    data = torch.load(str(ROOT / 'MuMoPepcan/data/processed_data/ligand_feats_PepDoRA_seq.pt'))
    labels = pd.read_excel(str(ROOT / 'MuMoPepcan/data/label/wet_exp.xlsx'), index_col='name')
    labels = labels.loc[list(data['name']),
                       ['animal', 'cell_100', 'cell_300', 'cell_1000']].values
    
    
    features = data['hidden_states'].mean(dim=1).numpy()  # [N, 384]
    
    
    idx_train = np.array([i for i, name in enumerate(data['name']) if not name.startswith('3')])
    idx_valid = np.array([i for i, name in enumerate(data['name']) if name.startswith('3')])
    
    return features, labels, idx_train, idx_valid

class SVMModel:
    def __init__(self):
        # SVM参数调优主要关注这些参数
        self.models = [SVR(
            kernel='poly',
            C=1.0,
            epsilon=0.1,
            tol=1e-3
        ) for _ in range(4)]
        self.scaler = StandardScaler()
        
    def train(self, X_train, y_train, **kwargs):
        X_train_scaled = self.scaler.fit_transform(X_train)
        for i, model in enumerate(self.models):
            print(f"Training SVM model {i+1}/4...")
            model.fit(X_train_scaled, y_train[:, i])
            
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        predictions = np.array([model.predict(X_scaled) for model in self.models]).T
        return predictions

class XGBoostModel:
    def __init__(self):
        self.models = [XGBRegressor(
            n_estimators=2000,
            learning_rate=0.09976304866708031,
            max_depth=25,
            min_child_weight=2,
            subsample=0.753748991862151,
            colsample_bytree=0.6678841070968727,
            random_state=3407
        ) for _ in range(4)]
        self.scaler = StandardScaler()
        
    def train(self, X_train, y_train, X_val, y_val):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        for i, model in enumerate(self.models):
            print(f"Training XGBoost model {i+1}/4...")
            
            eval_set = [(X_train_scaled, y_train[:, i]), 
                       (X_val_scaled, y_val[:, i])]
            model.fit(X_train_scaled, y_train[:, i],
                     eval_set=eval_set,
                     verbose=False)
            
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        predictions = np.array([model.predict(X_scaled) for model in self.models]).T
        return predictions

class RandomForestModel:
    def __init__(self):
        self.models = [RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=-1,
            random_state=3407
        ) for _ in range(4)]
        self.scaler = StandardScaler()
        
    def train(self, X_train, y_train, **kwargs):
        X_train_scaled = self.scaler.fit_transform(X_train)
        for i, model in enumerate(self.models):
            print(f"Training Random Forest model {i+1}/4...")
            model.fit(X_train_scaled, y_train[:, i])
            
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        predictions = np.array([model.predict(X_scaled) for model in self.models]).T
        return predictions


def train_and_evaluate(model_type='svm', k_fold=5):
    features, labels, idx_train, idx_valid = prepare_data()
    run_dir = str(ROOT / f'MuMoPepcan/runs/{model_type}_{get_fmt_time()}')
    os.makedirs(run_dir, exist_ok=True)
    
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=3407)
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(idx_train)):
        print(f"\nTraining fold {fold+1}/{k_fold}")
        
        
        X_train = features[idx_train[train_idx]]
        y_train = labels[idx_train[train_idx]]
        X_test = features[idx_train[test_idx]]
        y_test = labels[idx_train[test_idx]]
        X_valid = features[idx_valid]
        y_valid = labels[idx_valid]
        
        
        if model_type == 'svm':
            model = SVMModel()
            model.train(X_train, y_train)
        elif model_type == 'rf':
            model = RandomForestModel()
            model.train(X_train, y_train)
        else:
            model = XGBoostModel()
            model.train(X_train, y_train, X_test, y_test)

        
        test_pred = model.predict(X_test)
        valid_pred = model.predict(X_valid)
        
        
        test_mse = mean_squared_error(y_test, test_pred)
        valid_mse = mean_squared_error(y_valid, valid_pred)
        
        test_pr = np.mean([pearsonr(test_pred[:, i], y_test[:, i])[0] for i in range(4)])
        valid_pr = np.mean([pearsonr(valid_pred[:, i], y_valid[:, i])[0] for i in range(4)])
        
        fold_results.append({
            'fold': fold + 1,
            'test_loss': test_mse,
            'test_pr': test_pr,
            'valid_loss': valid_mse,
            'valid_pr': valid_pr
        })
        
        print(f"Fold {fold+1} - Test MSE: {test_mse:.4f}, Test PR: {test_pr:.4f}")
        print(f"Fold {fold+1} - Valid MSE: {valid_mse:.4f}, Valid PR: {valid_pr:.4f}")
    
    
    results_df = pd.DataFrame(fold_results)
    results_df.to_excel(os.path.join(run_dir, f'KFold_results.xlsx'), index=False)
    
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.boxplot([results_df['test_loss'], results_df['valid_loss']], labels=['Test MSE', 'Valid MSE'])
    plt.title(f'{model_type.upper()} - MSE Distribution')
    
    plt.subplot(1, 2, 2)
    plt.boxplot([results_df['test_pr'], results_df['valid_pr']], labels=['Test PR', 'Valid PR'])
    plt.title(f'{model_type.upper()} - Pearson Correlation Distribution')
    
    save_show(os.path.join(run_dir, f'{model_type}_performance.png'), 600, show=False)

if __name__ == '__main__':
    
    train_and_evaluate('svm', k_fold=5)
    
    
    train_and_evaluate('rf', k_fold=5)
    
    
    train_and_evaluate('xgboost', k_fold=5)