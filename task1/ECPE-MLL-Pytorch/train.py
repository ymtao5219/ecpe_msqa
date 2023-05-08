import torch
import argparse
from model import *
from utils import *
from dataset_loader import *
from config import * 
from loss import *

import ipdb 

def loss():
    pass

def train_loop():
    pass 

def inference(cml_out,eml_out,mode='avg'):
    eml_out_T = torch.permute(eml_out, (0,2,1))
    # print(eml_out_T.shape)
    if mode == 'avg':
        out = ((cml_out + eml_out_T)/2)>0.5
        out_ind = out.nonzero()
    elif mode == 'logic_and':
        cml_pair = cml_out>0.5
        eml_pair = eml_out>0.5
        out = torch.logical_and(cml_pair, eml_pair)
        out_ind = out.nonzero()
    elif mode == 'logic_or':
        cml_pair = cml_out>0.5
        eml_pair = eml_out>0.5
        out = torch.logical_or(cml_pair, eml_pair)
        out_ind = out.nonzero()

    return out_ind  # output index pairs: [batch,emo_clause,cause_clase]

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # doc_len_b: document length in a batch
    # adj_b: adj matrix in a batch, do not need this, #todo: will remove this
    # y_emotions_b: binary vector indicating emotion clause in a batch, -1 means no sentences in this document
    # y_causes_b: binary vector indicating cause clause in a batch, -1 means no sentences in this document
    # y_mask_b: binary vector indicating whether a sentence is valid in a batch, -1 means no sentences in this document
    # doc_couples_b: ground truth label in a batch
    # doc_id_b: document id in a batch
    # bert_token_b: input ids in a batch
    # bert_segment_b: segment ids in a batch
    # bert_masks_b: attention masks in a batch
    # bert_clause_b: [CLS] index for each doc in a batch
    doc_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, \
    bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b = instance
    

    bert_token_b = bert_token_b.to(device=device)
    bert_segment_b = bert_segment_b.to(device=device)
    bert_masks_b = bert_masks_b.to(device=device)
    bert_clause_b = bert_clause_b.to(device=device)
    
    sliding_mask = slidingmask_gen(D=configs.max_doc_len, W=configs.window_size, batch_size=configs.batch_size, device=device)
    
    configs = Config()
    train_set, val_set, test_set = load_data(configs)
    data_iter = iter(train_set)
    instance = next(data_iter)
    # instance = next(data_iter)

    
    prev_model = Network()
    prev_model = prev_model.to(device=device)

    prev_model.train()
    s1 = prev_model(bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b)
    s1 = input_padding(s1)
    s1.to(device=device)
    # print(s1.shape)
    # print(doc_couples_b)
    
    N = 2
    D = 75
    model = ISMLBlock(N, D, hidden_size)
    model = model.to(device=device)

    model.train()
    y_e_list, y_c_list, s_final, cml_scores, eml_scores = model(s1)
    ipdb.set_trace()
    # print(len(y_e_list),len(y_c_list),y_e_list[0].shape,y_c_list[0].shape)
    # print(s1)
    # print(s_final)
    # print(y_e_list)
    # print(y_c_list)
    ipdb.set_trace()
    loss_total,cml_out,eml_out = loss_calc(y_e_list[0].cpu(),y_c_list[0].cpu(),doc_couples_b,cml_scores.cpu(),eml_scores.cpu(),sliding_mask.cpu())
    print(loss_total)
    
    with torch.no_grad():
        model.eval()
        out_ind = inference(cml_out,eml_out,mode='avg')
        print(out_ind)

    # model.eval()
    # input_names = [ "actual_input" ]
    # output_names = [ "output" ]
    # torch.onnx.export(model, s1, 'secondhalf.onnx',input_names=input_names,\
    #              output_names=output_names,)

    # dots = torchviz.make_dot(s_final,params=dict(model.named_parameters()),show_attrs=False, show_saved=False)
    # dots.format = 'png'
    # dots.render('secondhalf_modelviz')

    # train_loop()
    
    # eval_loop()
    
    
    # print(model)
if __name__ == "__main__":
    configs = Config()
    main(configs)

