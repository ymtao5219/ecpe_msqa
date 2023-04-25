import torch
import argparse
from model import *
from utils import *
from dataset_loader import *
from config import * 

def train():
    pass 

def main(configs): 

    #############################
    # load data
    #############################
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
        valid_loader = build_inference_data(configs, fold_id=fold_id, data_type='valid')
        
    test_loader = build_inference_data(configs, fold_id=fold_id, data_type='test')
    #############################
    # train loop
    #############################
    for train_step, batch in enumerate(train_loader, 1):
        # todo: figure out each of the following variables
        doc_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, \
        bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b = batch
        
        print(batch)


    model = ISMLModel(configs.model_name, 
                      configs.n_hidden, 
                      configs.max_doc_len, 
                      configs.max_sen_len, 
                      configs.n_class, 
                      configs.model_type, 
                      configs.model_iter_num)
    
    print(model)
if __name__ == "__main__":
    config = Config()
    
    main(config)

