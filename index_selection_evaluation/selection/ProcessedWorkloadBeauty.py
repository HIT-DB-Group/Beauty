import os
import pickle
import numpy as np
import copy
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re
import time
from line_profiler import LineProfiler
from functools import wraps
import torch
import logging
import sys
import json

from selection import constants
from selection.workload import Workload,ProcessedWorkload,Plan
from selection.summary_cost_evaluation import CostEvaluation
from selection.index import Index
from selection.cost_evaluation import CostEvaluation as oringeCostEvaluation
from selection.utils import load_checkpoint,save2checkpoint


# 各个模型的类

from ML.AImeetsAI.model.AMA import aimeetsai as ama, aimeetsai_AE as ama_AE
from ML.AImeetsAI.model.AMA import aimeetsai_AE_MCD as ama_AE_MCD


from ML.AImeetsAI.dataset import Encoding as AMAEncoding
from ML.AImeetsAI.database_util import Batch as AMABatch,collator as AMACollator

ENCODING={
    'AMA':AMAEncoding,
    'AMA_autoencoder':AMAEncoding,
    'AMA_ensemble':AMAEncoding,
    'AMA_autoencoder_MCD':AMAEncoding,
}
BATCH={
    'AMA':AMABatch,
    'AMA_autoencoder':AMABatch,
    'AMA_ensemble':AMABatch,
    'AMA_autoencoder_MCD':AMABatch,

}
COLLATOR={
    'AMA':AMACollator,
    'AMA_autoencoder':AMACollator,
    'AMA_ensemble':AMACollator,
    'AMA_autoencoder_MCD':AMACollator,
}
MODEL={
    'AMA':ama,
    'AMA_autoencoder':ama_AE,
    'AMA_ensemble':ama,
    'AMA_autoencoder_MCD':ama_AE_MCD,
}

MODEL_DIR={
    'AMA':('ML/AImeetsAI','ama'),
    'AMA_autoencoder':('ML/AImeetsAI','ama_AE'),
    'AMA_ensemble':('ML/AImeetsAI','ama_ensemble'),
    'AMA_autoencoder_MCD':('ML/AImeetsAI','ama_AE_MCD'),
}



no_uncertainty=0
multi_inference_uncertainty=1
multi_model_uncertainty=2
autoencoder_uncertainty=3
multi_inference_autoencoder_uncertainty=4

estimator_classes={
    'AMA':no_uncertainty,
    'AMA_autoencoder':autoencoder_uncertainty,
    'AMA_ensemble':multi_model_uncertainty,
    'AMA_autoencoder_MCD':multi_inference_autoencoder_uncertainty,
}
AMA_NAIVE=2
WHAT_IF=3



class BeautyWorkload:
    def __init__(self,workload,cost_evaluation,parameters,config=constants.config):
        self.workload=workload
        self.parameters=parameters
        self.config=config
        self.cost_and_plan=workload.cost_and_plan
        self.benchmark_name=workload.benchmark_name
        self.cost_evaluation=cost_evaluation
        
        self.pWorkload=load_checkpoint(f'{constants.dir_path}/full_processed_workload')
        if torch.cuda.is_available():
            self.device='cuda'
        else:
            self.device='cpu'
        self.model_log=constants.model_log
        self._get_indexable_column_operator()
        self._get_naive_model()
        self._parse_plan()
        self._load_model()
        self.uncertainty_num=0
        self.all_inference_num=0

    def _get_indexable_column_operator(self):
        self.indexable_columns=self.pWorkload.indexable_columns
        self.indexable_columns_dict=self.pWorkload.indexable_columns_dict
        self.indexable_columns_name=self.pWorkload.indexable_columns_name
        self.indexable_table_columns_name=self.pWorkload.indexable_table_columns_name
    def _get_naive_model(self):
        model_name=self.parameters["estimator"].lower()
        if 'ama' in model_name:
            self.naive_model=AMA_NAIVE
        elif 'what_if' in model_name:
            self.naive_model=WHAT_IF
        else:
            raise Exception(f'no such naive model {model_name}')
    
    def _parse_plan(self):
        self.pWorkload.query2plan_dict={}
        for pplan in self.pWorkload.processed_plans:
            self.pWorkload.query2plan_dict[pplan.query.text]=pplan
        self.pplans=[]
        self.original_costs=[]
        for query in self.workload.queries:
            assert query.text in self.pWorkload.query2plan_dict.keys(), logging.info(f'query {query.text} not in pWorkload.query2plan_dict.keys()')
            pplan=self.pWorkload.query2plan_dict[query.text]
            self.pplans.append(pplan)
            self.original_costs.append(pplan.js_plans['Total Cost'])

    def _load_model(self):
        if self.naive_model==WHAT_IF:
            return
        self.crit=torch.nn.MSELoss(reduction='none')
        dir_info=MODEL_DIR[self.parameters["estimator"]]
        if self.model_log:
            self.model_dir=f'{dir_info[0]}/dataset/{self.config["benchmark_name"]}_log/model/{dir_info[1]}'
        else:
            self.model_dir=f'{dir_info[0]}/dataset/{self.config["benchmark_name"]}/model/{dir_info[1]}'

        if self.naive_model==AMA_NAIVE:
            if self.model_log:
                self.zero_idx=json.load(open(f'{dir_info[0]}/dataset/{self.config["benchmark_name"]}_log/data/zero_idx.json','r'))
            else:
                self.zero_idx=json.load(open(f'{dir_info[0]}/dataset/{self.config["benchmark_name"]}/data/zero_idx.json','r'))
            self.zero_idx=np.array(self.zero_idx['zero_idx'],dtype=np.bool)

        self.encoding=ENCODING[self.parameters["estimator"]](self.pWorkload)
        self.collator=COLLATOR[self.parameters["estimator"]]

        self.model_class=estimator_classes[self.parameters["estimator"]]

        self.model_config=json.load(open(f'{self.model_dir}/model_config.json','r'))
        self.uncertainty_threshold=self.model_config['uncertainty_threshold']

        if self.model_class!=multi_model_uncertainty:
            start=time.time()
            self.study=load_checkpoint(f'{self.model_dir}/study.pkl')
            load_trial_id=self.study.best_trial.number if "trial_id" not in self.model_config else self.model_config["trial_id"]
            mpd=self.study.trials[load_trial_id].params
            if self.naive_model==AMA_NAIVE:
                print(f'ama origin input_dims: {self.encoding.get_random_encoding()["features"].shape[0]}, drop zero_idx: {self.zero_idx.sum()}')
                mpd['input_dims']=self.encoding.get_random_encoding()['features'].shape[0]-self.zero_idx.sum()
            self.model = MODEL[self.parameters["estimator"]](mpd)
            self.model.to(self.device)
            self.model.load_state_dict(torch.load(f'{self.model_dir}/{load_trial_id}.pth',map_location=self.device))
            self.model.eval()
        else:
            start=time.time()
            self.study=load_checkpoint(f'{self.model_dir}/study.pkl')
            load_trial_ids=self.model_config["trial_id"]
            load_trial_ids=[int(i) for i in load_trial_ids.split('_')]
            self.models=[]
            for load_trial_id in load_trial_ids:
                mpd=self.study.trials[load_trial_id].params
                if self.naive_model==AMA_NAIVE:
                    print(f'ama origin input_dims: {self.encoding.get_random_encoding()["features"].shape[0]}, drop zero_idx: {self.zero_idx.sum()}')
                    mpd['input_dims']=self.encoding.get_random_encoding()['features'].shape[0]-self.zero_idx.sum()
                model = MODEL[self.parameters["estimator"]](mpd)
                model.to(self.device)
                model.load_state_dict(torch.load(f'{self.model_dir}/{load_trial_id}.pth',map_location=self.device))
                model.eval()
                self.models.append(model)

        

        

    def calculate_uncertainty(self,result_dict):
        need_whatif_list=torch.zeros(result_dict['batch_size'],dtype=torch.bool)
        
        if self.model_class == autoencoder_uncertainty or self.model_class==multi_inference_autoencoder_uncertainty:
            for k,v in result_dict['uncertainty'].items():
                if k=='query' or k=='index_attention':
                    assert len(v[0].shape)==3,print(f'query uncertainty uncertainty feature shape: {v[0].shape}')
                    if result_dict['seq_len'] is not None:
                        mask=torch.ones_like(v[0])
                        for i in range(len(result_dict['seq_len'])):
                            mask[i,result_dict['seq_len'][i]:,:]=0
                        uncertainty=self.crit(v[0],mask*v[1]).mean(dim=(1,2))
                    else:
                        uncertainty=self.crit(v[0],v[1]).mean(dim=(1,2))

                    threshold=self.uncertainty_threshold[k]
                    need_whatif_list[uncertainty>threshold]=True
                elif k=='y_uncertainty':
                    uncertainty=v
                    threshold=self.uncertainty_threshold[k]
                    need_whatif_list[uncertainty>threshold]=True
                else:
                    assert len(v[0].shape)==2,print(f'{k} uncertainty uncertainty feature shape: {v[0].shape}')
                    uncertainty=self.crit(v[0],v[1]).mean(dim=1)
                    threshold=self.uncertainty_threshold[k]
                    need_whatif_list[uncertainty>threshold]=True
        elif self.model_class==multi_model_uncertainty or self.model_class==multi_inference_uncertainty:
            for k,v in result_dict['uncertainty'].items():
                if k!='y_uncertainty':
                    raise NotImplementedError
                uncertainty=v
                threshold=self.uncertainty_threshold[k]
                need_whatif_list[uncertainty>threshold]=True
        else:
            raise NotImplementedError

        return need_whatif_list

    def model_inference(self,batch_dict,workload_list):
        '''
        no_uncertainty=0
        multi_inference_uncertainty=1
        multi_model_uncertainty=2
        autoencoder_uncertainty=3
        multi_inference_autoencoder_uncertainty=4
        '''
        batch_data=batch_dict['batch']
        batch_data=batch_data.to(self.device)
        seq_len=None if 'length' not in batch_dict else batch_dict['length']
        
        with torch.no_grad():

            if self.model_class==no_uncertainty:
                result_dict=self.model(batch_data)
                need_whatif_list=torch.zeros(len(workload_list),dtype=torch.bool)
            elif self.model_class==multi_inference_uncertainty:
                inference_time=20 if 'inference_time' not in self.model_config else self.model_config['inference_time']
                result_list=[]
                result_dict={}
                for _ in range(inference_time):
                    tem_result=self.model(batch_data)['output']
                    if len(tem_result.shape)==2:
                        tem_result=tem_result.squeeze(1)
                    result_list.append(tem_result)
                result_list=torch.stack(result_list)
                mean=torch.mean(result_list,dim=0)
                var=torch.var(result_list,dim=0)
                if mean.shape[0]==0:
                    mean=mean.unsqueeze(0)
                    var=var.unsqueeze(0)
                result_dict['output']=mean
                result_dict['uncertainty']={'y_uncertainty':var}
                result_dict['batch_size']=len(workload_list)
                need_whatif_list=self.calculate_uncertainty(result_dict)


            elif self.model_class==multi_model_uncertainty:
                result_list=[]
                result_dict={}
                for model in self.models:
                    tem_result=model(batch_data)['output']
                    if len(tem_result.shape)==2:
                        tem_result=tem_result.squeeze(1)
                    result_list.append(tem_result)
                result_list=torch.stack(result_list)
                mean=torch.mean(result_list,dim=0)
                var=torch.var(result_list,dim=0)
                if mean.shape[0]==0:
                    mean=mean.unsqueeze(0)
                    var=var.unsqueeze(0)
                result_dict['output']=mean
                result_dict['uncertainty']={'y_uncertainty':var}
                result_dict['batch_size']=len(workload_list)
                need_whatif_list=self.calculate_uncertainty(result_dict)


            elif self.model_class==autoencoder_uncertainty:
                result_dict=self.model(batch_data)
                result_dict['batch_size']=len(workload_list)
                result_dict['seq_len']=seq_len
                need_whatif_list=self.calculate_uncertainty(result_dict)
                

            elif self.model_class==multi_inference_autoencoder_uncertainty:
                inference_time=20 if 'inference_time' not in self.model_config else self.model_config['inference_time']
                result_list=[]
                for _ in range(inference_time):
                    result_dict=self.model(batch_data)
                    tem_result=result_dict['output']
                    if len(tem_result.shape)==2:
                        tem_result=tem_result.squeeze(1)
                    result_list.append(tem_result)
                result_list=torch.stack(result_list)
                mean=torch.mean(result_list,dim=0)
                var=torch.var(result_list,dim=0)
                if mean.shape[0]==0:
                    mean=mean.unsqueeze(0)
                    var=var.unsqueeze(0)
                result_dict['output']=mean
                result_dict['uncertainty']['y_uncertainty']=var
                result_dict['batch_size']=len(workload_list)
                result_dict['seq_len']=seq_len
                need_whatif_list=self.calculate_uncertainty(result_dict)


            model_cost_list=result_dict['output']
            if len(model_cost_list.shape)==2:
                model_cost_list=model_cost_list.squeeze(1)
            model_cost_list=model_cost_list.to('cpu')
        
        if self.model_log:
            model_cost_list=2-torch.exp(model_cost_list)

        return model_cost_list,need_whatif_list
    
    def get_cost_ml(self, workload_list, indexes):
        origin_costs=[self.original_costs[i] for i in workload_list]
        if len(indexes)==0:
            return origin_costs
        self.all_inference_num+=len(workload_list)
        origin_costs=torch.tensor(origin_costs)
        
        if self.naive_model==AMA_NAIVE:
            wk=Workload([self.workload.queries[i] for i in workload_list])
            what_if_cost_list,plan_list=self.cost_evaluation.calculate_cost_with_plan(wk, indexes, store_size=True)
            what_if_cost_list=torch.tensor(what_if_cost_list)
            feautres=[]
            for i,query_id in enumerate(workload_list):
                origin_feature=self.encoding.get_plan_encoder(self.pplans[query_id].js_plans)
                after_feature=self.encoding.get_plan_encoder(plan_list[i])
                feature=self.encoding.get_pair_features(origin_feature,after_feature)
                feature['features']=feature['features'][~self.zero_idx]
                feautres.append(feature)
            batch_dict=self.collator((feautres,0,0))
            model_cost_list,need_whatif_list=self.model_inference(batch_dict,workload_list)

            
            self.uncertainty_num+=need_whatif_list.sum()

            model_cost_list=(1-model_cost_list)*origin_costs
            
            model_cost_list[need_whatif_list]=what_if_cost_list[need_whatif_list]
            cost_list=model_cost_list

        elif self.naive_model==WHAT_IF:
            wk=Workload([self.workload.queries[i] for i in workload_list])
            what_if_cost_list=self.cost_evaluation.calculate_cost(wk, indexes, store_size=True)
            self.uncertainty_num+=len(workload_list)
            cost_list=torch.tensor(what_if_cost_list) 
        
        cost_list=cost_list.tolist()
        return cost_list

class TestSummary:
    def __init__(self, db_connector):
        self.db_connector=db_connector
        self.db_connector.drop_indexes()
        self.cost_evaluation=CostEvaluation(self.db_connector)
class GetCostEvaluation:
    def __init__(self, db_connector):
        self.db_connector=db_connector
        self.db_connector.drop_indexes()
        self.cost_evaluation=oringeCostEvaluation(self.db_connector)

