import torch
import argparse
from model import *
from utils import *
from dataset_loader import *
from config import * 

import ipdb 

def loss():
    pass

def train_loop():
    pass 

def eval_loop():
    pass

def load_data(configs):
    if configs.split == 'split10':
        n_folds = 10
        configs.epochs = 20
    elif configs.split == 'split20':
        n_folds = 20
        configs.epochs = 15
    else:
        print('Unknown data split.')
        exit()
        
    fold_id = 1
    train_loader = build_train_data(configs, fold_id=fold_id)
    if configs.split == 'split20':
        val_loader = build_inference_data(configs, fold_id=fold_id, data_type='valid')
        
    test_loader = build_inference_data(configs, fold_id=fold_id, data_type='test')
    return train_loader, val_loader, test_loader

def main(configs): 


    # load_data()
    train_set, val_set, test_set = load_data(configs)
    
    #################################################
    # for testing purpose
    #################################################
    data_iter = iter(train_set)
    instance = next(data_iter)

    doc_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, \
    bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b = instance
    
    # initialize model
    model = PretrainedBERT(configs.model_name)
    out = model(bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b)
    out2 = Pooler(100, 73)
    ipdb.set_trace()
    
    # train_loop()
    
    # eval_loop()
    
    
    # print(model)
if __name__ == "__main__":
    configs = Config()
    main(configs)

