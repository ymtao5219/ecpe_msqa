import torch
import argparse
from model import *
from utils import *
from dataset_loader import *
from config import * 
from loss import *

from transformers import get_linear_schedule_with_warmup
import torch.optim as optim

import ipdb 

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


def train_loop(configs, model, train_loader):

    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=configs.learning_rate, weight_decay=configs.weight_decay) 

    num_steps_all = len(train_loader) // configs.gradient_accumulation_steps * configs.EPOCHS
    warmup_steps = int(num_steps_all * configs.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps_all)

    optimizer.zero_grad()
    
    running_loss = 0.0
    for train_step, batch in enumerate(train_loader, 1):

        doc_len_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, \
        bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b = batch
        # ipdb.set_trace()
        
        bert_token_b = bert_token_b.to(device=device)
        bert_segment_b = bert_segment_b.to(device=device)
        bert_masks_b = bert_masks_b.to(device=device)
        bert_clause_b = bert_clause_b.to(device=device)
        
        sliding_mask = slidingmask_gen(D=configs.max_doc_len, 
                                        W=configs.window_size, 
                                        batch_size=configs.batch_size, 
                                        device=device)
        
        y_e_list, y_c_list, s_final, cml_scores, eml_scores = model(bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b)
        
        loss_total,cml_out,eml_out = loss_calc(y_e_list[0].cpu(),
                                                y_c_list[0].cpu(),
                                                doc_couples_b,
                                                cml_scores.cpu(),
                                                eml_scores.cpu(),
                                                sliding_mask.cpu())
        
        res = inference(cml_out, eml_out, mode='avg')
        # todo: calculate metrics
        # Backward pass
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss_total.item()

    return running_loss / len(train_loader)

def eval_loop(configs, model, val_loader):
    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        for val_step, batch in enumerate(val_loader, 1):
            doc_len_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, \
            bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b = batch
            # ipdb.set_trace()
            
            bert_token_b = bert_token_b.to(device=device)
            bert_segment_b = bert_segment_b.to(device=device)
            bert_masks_b = bert_masks_b.to(device=device)
            bert_clause_b = bert_clause_b.to(device=device)
            
            sliding_mask = slidingmask_gen(D=configs.max_doc_len, 
                                            W=configs.window_size, 
                                            batch_size=configs.batch_size, 
                                            device=device)
            
            y_e_list, y_c_list, s_final, cml_scores, eml_scores = model(bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b)
            
            loss_total,cml_out,eml_out = loss_calc(y_e_list[0].cpu(),
                                                    y_c_list[0].cpu(),
                                                    doc_couples_b,
                                                    cml_scores.cpu(),
                                                    eml_scores.cpu(),
                                                    sliding_mask.cpu())
            running_loss += loss_total.item()
            res = inference(cml_out, eml_out, mode='avg')
            # todo: calculate metrics
    return running_loss / len(val_loader)
        
def inference(cml_out, eml_out, mode='avg'):
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

def main(): 
    
    # load data 
    configs = Config()
    train_set, val_set, test_set = load_data(configs)

    EPOCHS = configs.EPOCHS
    
    # initilize the model
    model = Network(model_name="bert-base-chinese", 
                max_sen_len=configs.max_sen_len, 
                max_doc_len=configs.max_doc_len, 
                max_doc_len_bert=configs.max_doc_len_bert,
                model_iter_num=configs.model_iter_num, 
                window_size=configs.window_size, 
                n_hidden=configs.n_hidden, 
                n_class=configs.n_class).to(DEVICE)
    
    # train loop 
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_loop(configs, model, train_set)
        # Calculate average loss for the epoch
        val_loss = eval_loop(configs, model, val_set)
        print(f'Epoch {epoch}/{EPOCHS}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
    
    # test loop
    test_loss = eval_loop(configs, model, test_set)
    print(f'Test Loss: {test_loss:.4f}')

if __name__ == "__main__":

    main()

