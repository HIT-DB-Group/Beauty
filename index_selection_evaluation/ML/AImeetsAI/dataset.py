import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import json
import pandas as pd
import sys, os, copy,logging
from collections import deque
import time
from scipy.stats import pearsonr
import pickle
import random
import re

from ML.AImeetsAI.database_util import *
from selection import constants
from selection.utils import *
from selection.workload import ProcessedWorkload
from selection.index import Index


phy2log_map=constants.phy2log_map
indexed_type=constants.indexed_type
can_index_column_tpye=constants.can_index_column_tpye
can_index_operator_str=constants.can_index_operator_str


class Encoding:
    def __init__(self, pWorkload:ProcessedWorkload):
        self.pWorkload=pWorkload
        self.node_types2idx={'Unique':0,'Hash Join':1,'Bitmap Heap Scan':2,'Materialize':3,'SetOp':4,'Subquery Scan':5,'Aggregate':6,'BitmapAnd':7,'Gather Merge':8,'WindowAgg':9,'Sort':10,'Gather':11,'Index Scan':12,'Merge Join':13,'Bitmap Index Scan':14,'Nested Loop':15,'Index Only Scan':16,'CTE Scan':17,'Hash':18,'BitmapOr':19,'Limit':20,'Result':21,'Merge Append':22,'Append':23,'Group':24,'Seq Scan':25}

    def _parse_plan(self, plan:dict):
        node_id=self.node_types2idx[plan['Node Type']]
        parallel_aware=1 if plan['Parallel Aware'] else 0
        if 'Plans' not in plan.keys():
            if 'Actual Rows' in plan.keys():
                leaf_weight_bytes=plan['Actual Rows']*plan['Plan Width']
            else:
                leaf_weight_bytes=plan['Plan Rows']*plan['Plan Width']
            self.features[26*2+26*parallel_aware+node_id]=leaf_weight_bytes
            res=[(1,leaf_weight_bytes)]
        else:
            res=[]
            for sub_plan in plan['Plans']:
                res.extend(self._parse_plan(sub_plan))
            for height,leaf_weight_bytes in res:
                self.features[26*2+26*parallel_aware+node_id]+=height*leaf_weight_bytes
            res=[(height+1,leaf_weight_bytes) for height,leaf_weight_bytes in res]
    
        if 'Actual Startup Time' in plan.keys():
            self.features[26*parallel_aware+node_id]=plan['Actual Total Time']-plan['Actual Startup Time']
        else:
            self.features[26*parallel_aware+node_id]=plan['Total Cost']-plan['Startup Cost']
        
        
        return res

    def get_plan_encoder(self,js_plans):
        self.features=np.zeros(26*2*2)
        self._parse_plan(js_plans)
        return {'features':self.features}
    
    def normalize(self,features):
        value=np.sum(np.abs(features))
        if value==0:
            return features
        return features/value
    
    def get_pair_features(self,origin_features, indexed_feautres):
        features=indexed_feautres['features']-origin_features['features']
        for i in range(2):
            features[i*26*2:i*26*2+26*2]=self.normalize(features[i*26*2:i*26*2+26*2])
        return {'features':features}

    def get_random_encoding(self):
        features=np.random.randn(26*2*2)
        return {'features':features}


class LIBDataset(Dataset):
    def __init__(self, plan_cache:dict,query_text2query,split_path:str):
        super(LIBDataset, self).__init__()
        self.plan_cache=plan_cache
        self.queries_2_index_plan_dict={}
        self.queries_2_noindex_plan_dict={}

        self.collated_dicts=[]
        
        self.queries=[]
        self.indexes=[]
        self.plans=[]
        self.act_times=[]
        for k,v in plan_cache.items():
            self.queries.append(query_text2query[k[0]])
            self.indexes.append(k[1])
            self.plans.append(v)
            self.act_times.append(v['Actual Total Time'])
        self.pWorkload=ProcessedWorkload(constants.config)
        self.pWorkload.add_from_triple(self.queries,self.indexes,self.plans)
        
        if type(list(self.indexes[0])[0])==str:
            print(f'indexes is str, need to convert to Index')
            indexes=[]
            for index in self.indexes:
                indexes.append(self._indexes_str2indexes(index))
            self.indexes=indexes
            self.pWorkload.indexes=indexes
        else :
            print(f'indexes is Index, no need to convert to Index')
        
        self.split_data_info=load_checkpoint(split_path)
        
        self.encoding=Encoding(self.pWorkload)

        for query,index,pPlan,act_time in zip(self.queries,self.indexes,self.pWorkload.processed_plans,self.act_times):
            plan_feature=self.encoding.get_plan_encoder(pPlan.js_plans)
            
            if query in self.queries_2_index_plan_dict:
                self.queries_2_index_plan_dict[query].append((index,plan_feature,act_time))
            else:
                self.queries_2_index_plan_dict[query]=[(index,plan_feature,act_time)]
            if len(index)==0:
                self.queries_2_noindex_plan_dict[query]=(index,plan_feature,act_time)
        

    def generate_data_with_OOD(self,drop_threshold=None):

        self.data_collated_dicts={'train':[],'test':[],'ood':[],'random':[]}
        self.data_collated_info_dicts={'train':[],'test':[],'ood':[],'random':[]}

        for query,ips in self.queries_2_index_plan_dict.items():
            no_index,no_plan_feature,no_time=self.queries_2_noindex_plan_dict[query]
            for ip in ips:
                collated=self.encoding.get_pair_features(no_plan_feature,ip[1])
                ratio=(no_time-ip[2])/no_time
                real=ip[2]
                if drop_threshold is not None and ratio<drop_threshold:
                    continue
                classes=self._get_classes(query,ip[0])
                self.data_collated_dicts[classes].append((collated,ratio,real))
                self.data_collated_info_dicts[classes].append((query,ip[0]))
                
                self.collated_dicts.append((collated,ratio,real))
        
        for _ in range(100):
            collated=self.encoding.get_random_encoding()
            self.data_collated_dicts['random'].append((collated,0,0))
            self.data_collated_info_dicts['random'].append((None,None))
            self.collated_dicts.append((collated,None,None))
        for k,v in self.data_collated_dicts.items():
            print(f'{k} has {len(v)} data')
        return self.data_collated_dicts
    
    def _get_classes(self,query,index):
        if query.nr in self.split_data_info['ood_query']:
            return 'ood'
        elif (query.nr,self._frozenset_to_str(index)) in self.split_data_info['test_query_index']:
            return 'test'
        else:
            return 'train'
        
    def _indexes_str2indexes(self,indexes_str_list):
        indexes_config=[]
        for indexes_str in indexes_str_list:
            columns_str=re.findall(r'\.(\w+)',indexes_str)
            columns=[]
            for column_str in columns_str:
                i=self.pWorkload.indexable_columns_dict[column_str]
                columns.append(self.pWorkload.indexable_columns[i])
            indexes_config.append(Index(columns))
            
        return frozenset(indexes_config)

    def _frozenset_to_str(self,index_frozenset):
        indexes=[str(index) for index in index_frozenset]
        indexes.sort()
        return ','.join(indexes)
    
    def shuffle(self):
        for k,v in self.data_collated_dicts.items():
            seed=random.randint(0,100)
            np.random.seed(seed)
            np.random.shuffle(v)
            np.random.seed(seed)
            np.random.shuffle(self.data_collated_info_dicts[k])

    
    def dump_collated_dicts(self,path):
        save2checkpoint(obj=self.data_collated_dicts,path=path)
        save2checkpoint(obj=self.data_collated_info_dicts,path=path[:-4]+'_info'+path[-4:])




class MiddleDataset(Dataset):
    def __init__(self, collated_dicts:list):
        self.collated_dicts=collated_dicts
    
    def __len__(self):
        return len(self.collated_dicts)
    
    def __getitem__(self, idx):
        
        return self.collated_dicts[idx][0], self.collated_dicts[idx][1],self.collated_dicts[idx][2]

def get_ts_vs(path,train_ratio):
    collated_dicts=load_checkpoint(path)
    train_len=int(train_ratio*len(collated_dicts))
    return MiddleDataset(collated_dicts[:train_len]),MiddleDataset(collated_dicts[train_len:])


def get_four_ts_vs(path,train_ratio,drop_zero=False):
    data_collated_dicts=load_checkpoint(path)
    train_len=int(train_ratio*len(data_collated_dicts['both_seen']))
    md_dict={}
    md_dict['both_seen_train']=MiddleDataset(data_collated_dicts['both_seen'][:train_len])
    md_dict['both_seen_test']=MiddleDataset(data_collated_dicts['both_seen'][train_len:])
    md_dict['both_seen']=MiddleDataset(data_collated_dicts['both_seen'])
    md_dict['query_seen']=MiddleDataset(data_collated_dicts['query_seen'])
    md_dict['index_seen']=MiddleDataset(data_collated_dicts['index_seen'])
    md_dict['both_unseen']=MiddleDataset(data_collated_dicts['both_unseen'])
    md_dict['random']=MiddleDataset(data_collated_dicts['random'])

    keys=['both_seen_train','both_seen_test','both_seen','query_seen','index_seen','both_unseen']
    total_features=[]
    for k in keys:
        total_features.extend([d[0]['features'] for d in md_dict[k]])

    total_features=np.array(total_features)
    total_dims=total_features.shape[-1]
    
    if drop_zero:

        zero_idx=np.abs(total_features).sum(axis=0)==0
        print(f'zero_idx len {zero_idx.sum()}')
        for k,v in md_dict.items():
            if k=='both_seen_train' or k=='both_seen_test':
                continue
            for i in range(len(v)):
                v[i][0]['features']=v[i][0]['features'][~zero_idx]
        return md_dict,total_dims-zero_idx.sum()
    else:
        return md_dict,total_dims

def get_prepare_ts_vs(path,drop_zero=False):
    data_collated_dicts=load_checkpoint(path)
    md_dict={}
    for k,v in data_collated_dicts.items():
        md_dict[k]=MiddleDataset(v)
    
    keys=data_collated_dicts.keys()
    total_features=[]
    for k in keys:
        if k=='random':
            continue
        total_features.extend([d[0]['features'] for d in md_dict[k]])

    total_features=np.array(total_features)
    total_dims=total_features.shape[-1]
    if drop_zero:

        zero_idx=np.abs(total_features).sum(axis=0)==0
        print(f'zero_idx len {zero_idx.sum()}')
        for k,v in md_dict.items():
            for i in range(len(v)):
                v[i][0]['features']=v[i][0]['features'][~zero_idx]
        return md_dict,total_dims-zero_idx.sum()
    else:
        return md_dict,total_dims

