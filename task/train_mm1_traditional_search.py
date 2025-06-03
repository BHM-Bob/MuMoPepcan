import os
import platform
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from mbapy.base import get_fmt_time
from mbapy.plot import save_show
from scipy.stats import pearsonr, randint, uniform
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
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
        self.models = []
        self.scaler = StandardScaler()
        
        self.param_distributions = {
            'kernel': ['rbf', 'linear', 'poly'],
            'C': uniform(0.1, 10.0),
            'epsilon': uniform(0.01, 0.2),
            'gamma': ['scale', 'auto'] + list(uniform(0.001, 0.1).rvs(10))
        }
        
    def train(self, X_train, y_train, n_iter=20, **kwargs):
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        for i in range(4):
            print(f"\nOptimizing SVM model {i+1}/4...")
            base_model = SVR(tol=1e-3)
            
            
            random_search = RandomizedSearchCV(
                base_model,
                param_distributions=self.param_distributions,
                n_iter=n_iter,
                cv=5,
                random_state=3407,
                n_jobs=-1,
                verbose=1
            )
            
            
            random_search.fit(X_train_scaled, y_train[:, i])
            print(f"Best parameters for model {i+1}: {random_search.best_params_}")
            print(f"Best CV score: {random_search.best_score_:.4f}")
            
            self.models.append(random_search.best_estimator_)
            
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        predictions = np.array([model.predict(X_scaled) for model in self.models]).T
        return predictions
    

class XGBoostModel:
    def __init__(self):
        self.models = []
        self.scaler = StandardScaler()
        
    def _objective(self, trial, X_train, y_train, X_val, y_val):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'random_state': 3407
        }
        
        model = XGBRegressor(**param)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        
        val_pred = model.predict(X_val)
        return mean_squared_error(y_val, val_pred)
    
    def train(self, X_train, y_train, X_val, y_val, n_trials=100):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        for i in range(4):  # 4个目标
            print(f"\nOptimizing XGBoost model {i+1}/4...")
            
            
            study = optuna.create_study(direction='minimize')
            
            
            study.optimize(
                lambda trial: self._objective(
                    trial, 
                    X_train_scaled, y_train[:, i],
                    X_val_scaled, y_val[:, i]
                ),
                n_trials=n_trials,
                show_progress_bar=True
            )
            
            print(f"Best parameters for model {i+1}: {study.best_params}")
            print(f"Best MSE: {study.best_value:.4f}")
            
            
            best_model = XGBRegressor(**study.best_params)
            best_model.fit(
                X_train_scaled, y_train[:, i],
                eval_set=[(X_val_scaled, y_val[:, i])],
                verbose=False
            )
            self.models.append(best_model)
            
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        predictions = np.array([model.predict(X_scaled) for model in self.models]).T
        return predictions
    

def train_and_evaluate(model_type='svm', k_fold=5):
    features, labels, idx_train, idx_valid = prepare_data()
    run_dir = str(ROOT / f'MuMoPepcan/runs/{model_type}_{get_fmt_time()}')
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(__file__, run_dir)
    
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
            model.train(X_train, y_train, n_iter=500)
        else:
            model = XGBoostModel()
            model.train(X_train, y_train, X_test, y_test, n_trials=500)
        
        
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
    results_df.to_excel(os.path.join(run_dir, f'{model_type}_results.xlsx'), index=False)
    
    
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
    
    train_and_evaluate('xgboost', k_fold=5)