'''
Date: 2025-04-04 10:46:55
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-05-25 17:17:14
Description: 
'''
import os
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
from mbapy.plot import save_show


def save_model_dict(model, model_dir, msg):
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    print("model has been saved to %s." % (model_path))

def load_model_dict(model, ckpt):
    model.load_state_dict(torch.load(ckpt))

def write_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

class BestMeter(object):
    """Computes and stores the best value"""

    def __init__(self, best_type):
        self.best_type = best_type  
        self.count = 0 
        self.sum_best = 0     
        self.reset()

    def reset(self):
        self.sum_best = 0
        if self.best_type == 'min':
            self.best = float('inf')
        else:
            self.best = -float('inf')

    def update(self, best):
        """set the best value, reset the counter to 0, add 1 to the sum_best"""
        self.best = best
        self.count = 0
        self.sum_best += 1

    def get_best(self):
        return self.best

    def counter(self):
        """add 1 to the counter, and return the counter value"""
        self.count += 1
        return self.count


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg


def draw_loss(train_ls: list[float], test_ls: list[float], valid_ls: list[float], run_dir: str, fold: int):
    plt.figure(figsize=(14, 8))
    plt.plot(train_ls, label='train')
    plt.plot(test_ls, label='test')
    plt.plot(valid_ls, label='valid')
    plt.legend()
    save_show(os.path.join(run_dir, f'loss_{fold}.png'), 600, show=False)
    plt.ylim(0, 3)
    save_show(os.path.join(run_dir, f'loss_zoom_{fold}.png'), 600, show=False)


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total_num, 'trainable': trainable_num}




