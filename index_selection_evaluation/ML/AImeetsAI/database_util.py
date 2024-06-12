import numpy as np
import pandas as pd
import csv
import torch
import torch.nn.functional as F
import torch.nn as nn
import time


class Batch():
    def __init__(self, features):
        super(Batch, self).__init__()

        self.features = features
        
    def to(self, device):

        self.features = self.features.to(device)
        return self
    
    def __len__(self):
        return self.features.shape[0]



def collator(small_set,drop_zero=False,log_process=False):
    if drop_zero:
        small_set[0]=list(small_set[0])
        small_set[1]=list(small_set[1])
        small_set[2]=list(small_set[2])
        to_drop=[]
        for i,s in enumerate(small_set[0]):
            if len(s['features'])==0 or np.abs(s['features']).sum()<1e-6:
                to_drop.append(i)
        if len(to_drop)==len(small_set[0]):
            to_drop=to_drop[:-1]
        for i in to_drop[::-1]:
            small_set[0].pop(i)
            small_set[1].pop(i)
            small_set[2].pop(i)
    y_ratios = small_set[1]
    y_reals=small_set[2]
    
    if log_process:
        y_ratios=torch.tensor(y_ratios, dtype=torch.float)
        y_ratios=torch.log(2-y_ratios)
    else:
        y_ratios=torch.tensor(y_ratios, dtype=torch.float)


    features=np.array([s['features'] for s in small_set[0]])
    features=torch.tensor(np.stack(features), dtype=torch.float)

    bth=Batch(features)
    batch_dict={
        'batch':bth,
        'y_ratios':torch.tensor(y_ratios, dtype=torch.float),
        'y_reals':torch.tensor(y_reals, dtype=torch.float),
    }
    return batch_dict


