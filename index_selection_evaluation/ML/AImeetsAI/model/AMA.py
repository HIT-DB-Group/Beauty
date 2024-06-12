import torch
from torch.utils.data import Dataset
import json
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import torchbnn as bnn
from alpaca.uncertainty_estimator.masks import build_mask


HIDDEN_DIMS=[[128,128], [128,128,64], [128,64,32], [128,64], [256,128], [256,128,64],[64],[128],[256]]
FINAL_DIMS=[[],[1],[0.5],[1,0.5],[1,1],[1,0.5,0.25]]
ACTIVATION_FUNC = [nn.PReLU(),nn.Tanh()]
FINAL_ACTIVATION_FUNC = [nn.PReLU(),None]

class aimeetsai(nn.Module):
    def __init__(self, config):
        super(aimeetsai, self).__init__()
        input_dims = 156 if 'input_dims' not in config else config['input_dims']
        hidden_dims = [128,128] if 'hidden_dims' not in config else HIDDEN_DIMS[config['hidden_dims']]
        dropout = 0.1 if 'dropout' not in config else config['dropout']
        activation_function=ACTIVATION_FUNC[0] if 'activation_function' not in config else ACTIVATION_FUNC[config['activation_function']]
        final_activation_funciton=FINAL_ACTIVATION_FUNC[0] if 'final_activation_funciton' not in config else FINAL_ACTIVATION_FUNC[config['final_activation_funciton']]

        self.layers=nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i==0:
                self.layers.append(nn.Linear(input_dims, hidden_dims[i]))
            else:
                self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.layers.append(activation_function)
            self.layers.append(nn.Dropout(dropout))

        self.layers.append(nn.Linear(hidden_dims[-1], 1))
        if final_activation_funciton is not None:
            self.layers.append(final_activation_funciton)


    def forward(self, batched_data):
        features=batched_data.features
        for layer in self.layers:
            features=layer(features)
        return {'output':features}


class aimeetsai_AE(nn.Module):
    def __init__(self, config):
        super(aimeetsai_AE, self).__init__()
        input_dims = 156 if 'input_dims' not in config else config['input_dims']
        hidden_dims = [128,128] if 'hidden_dims' not in config else HIDDEN_DIMS[config['hidden_dims']]
        dropout = 0.1 if 'dropout' not in config else config['dropout']
        final_dims=[] if 'final_dims' not in config else FINAL_DIMS[config['final_dims']]
        activation_function=ACTIVATION_FUNC[0] if 'activation_function' not in config else ACTIVATION_FUNC[config['activation_function']]
        final_activation_funciton=FINAL_ACTIVATION_FUNC[0] if 'final_activation_funciton' not in config else FINAL_ACTIVATION_FUNC[config['final_activation_funciton']]

        self.encoders=nn.ModuleList()
        self.decoders=nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i==0:
                self.encoders.append(nn.Linear(input_dims, hidden_dims[i]))
            else:
                self.encoders.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.encoders.append(activation_function)
            self.encoders.append(nn.Dropout(dropout))

            self.decoders.insert(0, nn.Dropout(dropout))
            self.decoders.insert(0, activation_function)
            if i==0:
                self.decoders.insert(0, nn.Linear(hidden_dims[i], input_dims))
            else:
                self.decoders.insert(0, nn.Linear(hidden_dims[i], hidden_dims[i-1]))
        
        encoder_final_dim=hidden_dims[-1]
        final_dim=encoder_final_dim
        self.preds=nn.ModuleList()
        for i in range(len(final_dims)):
            if i==0:
                self.preds.append(nn.Linear(encoder_final_dim, int(encoder_final_dim*final_dims[i])))
            else:
                self.preds.append(nn.Linear(int(encoder_final_dim*final_dims[i-1]), int(encoder_final_dim*final_dims[i])))
            final_dim=int(encoder_final_dim*final_dims[i])
            self.preds.append(activation_function)
            self.preds.append(nn.Dropout(dropout))

        self.preds.append(nn.Linear(final_dim, 1))
        if final_activation_funciton is not None:
            self.preds.append(final_activation_funciton)

    def forward(self, batched_data):
        features=batched_data.features
        origin_features=copy.deepcopy(features)
        for layer in self.encoders:
            features=layer(features)
        latent_features=features.clone()
        for layer in self.decoders:
            latent_features=layer(latent_features)
        
        
        for layer in self.preds:
            features=layer(features)
        result_dict={
            'output':features,
            'uncertainty':{
                'index_query':(origin_features,latent_features)
            }
        }
        return result_dict
    
    def forward_uncertainty(self, batched_data):
        features=batched_data.features
        origin_features=copy.deepcopy(features)
        for layer in self.encoders:
            features=layer(features)
        latent_features=features.clone()
        for layer in self.decoders:
            latent_features=layer(latent_features)
        
        return {'uncertainty':{'index_query':(origin_features,latent_features)}}

    def lock_encoder(self):
        for param in self.encoders.parameters():
            param.requires_grad = False

class aimeetsai_AE_MCD(nn.Module):
    def __init__(self, config):
        super(aimeetsai_AE_MCD, self).__init__()
        input_dims = 156 if 'input_dims' not in config else config['input_dims']
        hidden_dims = [128,128] if 'hidden_dims' not in config else HIDDEN_DIMS[config['hidden_dims']]
        dropout = 0.1 if 'dropout' not in config else config['dropout']
        final_dims=[] if 'final_dims' not in config else FINAL_DIMS[config['final_dims']]
        activation_function=ACTIVATION_FUNC[0] if 'activation_function' not in config else ACTIVATION_FUNC[config['activation_function']]
        final_activation_funciton=FINAL_ACTIVATION_FUNC[0] if 'final_activation_funciton' not in config else FINAL_ACTIVATION_FUNC[config['final_activation_funciton']]

        self.encoders=nn.ModuleList()
        self.decoders=nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i==0:
                self.encoders.append(nn.Linear(input_dims, hidden_dims[i]))
            else:
                self.encoders.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.encoders.append(activation_function)
            self.encoders.append(nn.Dropout(dropout))

            self.decoders.insert(0, nn.Dropout(dropout))
            self.decoders.insert(0, activation_function)
            if i==0:
                self.decoders.insert(0, nn.Linear(hidden_dims[i], input_dims))
            else:
                self.decoders.insert(0, nn.Linear(hidden_dims[i], hidden_dims[i-1]))
        
        encoder_final_dim=hidden_dims[-1]
        final_dim=encoder_final_dim
        self.preds=nn.ModuleList()
        for i in range(len(final_dims)):
            if i==0:
                self.preds.append(nn.Linear(encoder_final_dim, int(encoder_final_dim*final_dims[i])))
            else:
                self.preds.append(nn.Linear(int(encoder_final_dim*final_dims[i-1]), int(encoder_final_dim*final_dims[i])))
            final_dim=int(encoder_final_dim*final_dims[i])
            self.preds.append(activation_function)
            self.preds.append(DropoutMC(dropout,activate=True))

        self.preds.append(nn.Linear(final_dim, 1))
        if final_activation_funciton is not None:
            self.preds.append(final_activation_funciton)

    def forward(self, batched_data):
        features=batched_data.features
        origin_features=copy.deepcopy(features)
        for layer in self.encoders:
            features=layer(features)
        latent_features=features.clone()
        for layer in self.decoders:
            latent_features=layer(latent_features)
        
        
        for layer in self.preds:
            features=layer(features)
        
        result_dict={
            'output':features,
            'uncertainty':{
                'index_query':(origin_features,latent_features)
            }
        }
        return result_dict
    
    def forward_uncertainty(self, batched_data):
        features=batched_data.features
        origin_features=copy.deepcopy(features)
        for layer in self.encoders:
            features=layer(features)
        latent_features=features.clone()
        for layer in self.decoders:
            latent_features=layer(latent_features)

        return {'uncertainty':{'index_query':(origin_features,latent_features)}}

    def lock_encoder(self):
        for param in self.encoders.parameters():
            param.requires_grad = False


class aimeetsai_BNN(nn.Module):
    def __init__(self, config):
        super(aimeetsai_BNN, self).__init__()
        input_dims = 156 if 'input_dims' not in config else config['input_dims']
        hidden_dims = [128,128] if 'hidden_dims' not in config else HIDDEN_DIMS[config['hidden_dims']]
        dropout = 0.1 if 'dropout' not in config else config['dropout']
        activation_function=ACTIVATION_FUNC[0] if 'activation_function' not in config else ACTIVATION_FUNC[config['activation_function']]
        final_activation_funciton=FINAL_ACTIVATION_FUNC[0] if 'final_activation_funciton' not in config else FINAL_ACTIVATION_FUNC[config['final_activation_funciton']]

        self.layers=nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i==0:
                self.layers.append(bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=input_dims, out_features=hidden_dims[i]))
            else:
                self.layers.append(bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_dims[i-1], out_features=hidden_dims[i]))
            self.layers.append(activation_function)
            self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(hidden_dims[-1], 1))
        if final_activation_funciton is not None:
            self.layers.append(final_activation_funciton)

    def forward(self, batched_data):
        features=batched_data.features
        for layer in self.layers:
            features=layer(features)
        return {'output':features}


class aimeetsai_MCD_all(nn.Module):
    def __init__(self, config):
        super(aimeetsai_MCD_all, self).__init__()
        input_dims = 156 if 'input_dims' not in config else config['input_dims']
        hidden_dims = [128,128] if 'hidden_dims' not in config else HIDDEN_DIMS[config['hidden_dims']]
        dropout = 0.1 if 'dropout' not in config else config['dropout']
        activation_function=ACTIVATION_FUNC[0] if 'activation_function' not in config else ACTIVATION_FUNC[config['activation_function']]
        final_activation_funciton=FINAL_ACTIVATION_FUNC[0] if 'final_activation_funciton' not in config else FINAL_ACTIVATION_FUNC[config['final_activation_funciton']]

        self.layers=nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i==0:
                self.layers.append(nn.Linear(input_dims, hidden_dims[i]))
            else:
                self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.layers.append(activation_function)
            self.layers.append(DropoutMC(dropout,activate=True))

        self.layers.append(nn.Linear(hidden_dims[-1], 1))
        if final_activation_funciton is not None:
            self.layers.append(final_activation_funciton)
        

    def forward(self, batched_data):
        features=batched_data.features
        for layer in self.layers:
            features=layer(features)
        return {'output':features}

class aimeetsai_MCD_part(nn.Module):
    def __init__(self, config):
        super(aimeetsai_MCD_part, self).__init__()
        input_dims = 156 if 'input_dims' not in config else config['input_dims']
        hidden_dims = [128,128] if 'hidden_dims' not in config else HIDDEN_DIMS[config['hidden_dims']]
        dropout = 0.1 if 'dropout' not in config else config['dropout']
        activation_function=ACTIVATION_FUNC[0] if 'activation_function' not in config else ACTIVATION_FUNC[config['activation_function']]
        final_activation_funciton=FINAL_ACTIVATION_FUNC[0] if 'final_activation_funciton' not in config else FINAL_ACTIVATION_FUNC[config['final_activation_funciton']]

        self.layers=nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i==0:
                self.layers.append(nn.Linear(input_dims, hidden_dims[i]))
            else:
                self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.layers.append(activation_function)
            if i==len(hidden_dims)-1:
                self.layers.append(DropoutMC(dropout,activate=True))
            else:
                self.layers.append(DropoutMC(dropout,activate=False))

        self.layers.append(nn.Linear(hidden_dims[-1], 1))
        if final_activation_funciton is not None:
            self.layers.append(final_activation_funciton)
        

    def forward(self, batched_data):
        features=batched_data.features
        for layer in self.layers:
            features=layer(features)
        return {'output':features}

class aimeetsai_MCD_dpp(nn.Module):
    def __init__(self, config):
        super(aimeetsai_MCD_dpp, self).__init__()
        input_dims = 156 if 'input_dims' not in config else config['input_dims']
        hidden_dims = [128,128] if 'hidden_dims' not in config else HIDDEN_DIMS[config['hidden_dims']]
        dropout = 0.1 if 'dropout' not in config else config['dropout']
        activation_function=ACTIVATION_FUNC[0] if 'activation_function' not in config else ACTIVATION_FUNC[config['activation_function']]
        final_activation_funciton=FINAL_ACTIVATION_FUNC[0] if 'final_activation_funciton' not in config else FINAL_ACTIVATION_FUNC[config['final_activation_funciton']]

        self.layers=nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i==0:
                self.layers.append(nn.Linear(input_dims, hidden_dims[i]))
            else:
                self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.layers.append(activation_function)
            if i==len(hidden_dims)-1:
                self.layers.append(DropoutDPP(dropout,activate=True))
            else:
                self.layers.append(DropoutMC(dropout,activate=False))

        self.layers.append(nn.Linear(hidden_dims[-1], 1))
        if final_activation_funciton is not None:
            self.layers.append(final_activation_funciton)
        

    def forward(self, batched_data):
        features=batched_data.features
        for layer in self.layers:
            features=layer(features)
        return {'output':features}




class DropoutMC(torch.nn.Module):
    def __init__(self, p: float, activate=False):
        super().__init__()
        self.activate = activate
        self.p = p
        self.p_init = p

    def forward(self, x: torch.Tensor):
        # self.training在model.train()时为true，eval()为false
        return torch.nn.functional.dropout(
            x, self.p, training=self.training or self.activate
        )
    
def reset_all_dpp_dropouts(model):
    for layer in model.children():
        if isinstance(layer, DropoutDPP):
            layer.mask.reset()
        else:
            reset_all_dpp_dropouts(model=layer)


class DropoutDPP(DropoutMC):
    dropout_id = -1

    def __init__(
        self,
        p: float,
        activate=False,
        mask_name="dpp",
        max_n=100,
        max_frac=0.4,
        coef=1.0,
    ):
        super().__init__(p=p, activate=activate)

        self.mask = build_mask(mask_name)
        self.reset_mask = False
        self.max_n = max_n
        self.max_frac = max_frac
        self.coef = coef

        self.curr_dropout_id = DropoutDPP.update()
        print(f"Dropout id: {self.curr_dropout_id}")

    @classmethod
    def update(cls):
        cls.dropout_id += 1
        return cls.dropout_id

    def calc_mask(self, x: torch.Tensor):
        return self.mask(x, dropout_rate=self.p, layer_num=self.curr_dropout_id).float()

    def get_mask(self, x: torch.Tensor):
        return self.mask(x, dropout_rate=self.p, layer_num=self.curr_dropout_id).float()


    def calc_non_zero_neurons(self, sum_mask):
        # assert 
        try:
            frac_nonzero = (sum_mask != 0).sum(axis=-1).item() / sum_mask.shape[-1]
        except Exception as e:
            print(f'sum_mask.shape:{sum_mask.shape}')
            print(f'sum_mask tpye:{type(sum_mask)}')
            print(f'sum_mask.device:{sum_mask.device}')
            print(f'sum_mask:{sum_mask}')
            print(e)

        return frac_nonzero

    def forward(self, x: torch.Tensor):
        # print(f'在dpp的forward中,requires_grad {x.requires_grad}')
        if self.training:
            return torch.nn.functional.dropout(x, self.p, training=True)
        else:
            if not self.activate:
                return x

            sum_mask = self.get_mask(x)

            norm = 1.0
            i = 1
            frac_nonzero = self.calc_non_zero_neurons(sum_mask)
            # print('==========Non zero neurons:', frac_nonzero, 'iter:', i, 'id:', self.curr_dropout_id, '******************')
            # while i < 30:
            while i < self.max_n and frac_nonzero < self.max_frac:
                # while frac_nonzero < self.max_frac:
                mask = self.get_mask(x)

                # sum_mask = self.coef * sum_mask + mask
                sum_mask += mask
                i += 1
                # norm = self.coef * norm + 1

                frac_nonzero = self.calc_non_zero_neurons(sum_mask)
                # print('==========Non zero neurons:', frac_nonzero, 'iter:', i, '******************')

            # res = x * sum_mask / norm
            # print("Number of masks:", i)
            if sum_mask.device != x.device:
                sum_mask = sum_mask.to(x.device)
            res = x * sum_mask / i
            return res
