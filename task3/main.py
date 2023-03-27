import sys, os, warnings, time
sys.path.append('..')
warnings.filterwarnings("ignore")
import numpy as np
import torch
from config import *
from data_loader import *


torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)

configs = Config()

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

for train_step, batch in enumerate(train_loader, 1):
    # todo: figure out each of the following variables
    doc_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, \
    bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b = batch
    
    print(batch)
