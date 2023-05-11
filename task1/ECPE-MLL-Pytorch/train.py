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

import tqdm

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
        
    fold_id = 2
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
    tp_epoch,predict_len_epoch,gt_len_epoch = 0,0,0
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
        with torch.no_grad():
            res = inference(cml_out, eml_out, mode='avg')
            # todo: calculate metrics
            tp,predict_len,gt_len = calculate_metrics(doc_couples_b, res, y_mask_b)
            
            # tp,predict_len,gt_len = check_accuracy_batch(doc_couples_b,res)
            # print(tp,predict_len,gt_len)
            # ipdb.set_trace()

            tp_epoch += tp
            predict_len_epoch += predict_len
            gt_len_epoch += gt_len
        
        # Backward pass
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss_total.item()

        # print(batch)

    with torch.no_grad():
        # metrics
        # precision,recall,f1 = metrics_calc(tp_epoch,predict_len_epoch,gt_len_epoch)
        precision, recall, f1 = metrics_calc(tp_epoch,predict_len_epoch,gt_len_epoch)
        # print(precision,recall,f1)
        # ipdb.set_trace()

    return running_loss / len(train_loader),precision,recall,f1

def eval_loop(configs, model, val_loader):
    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        tp_epoch,predict_len_epoch,gt_len_epoch = 0,0,0
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
            tp,predict_len,gt_len = calculate_metrics(doc_couples_b, res, y_mask_b)
            # todo: calculate metrics
        #     tp,predict_len,gt_len = check_accuracy_batch(doc_couples_b,res)
            tp_epoch += tp
            predict_len_epoch += predict_len
            gt_len_epoch += gt_len

        precision,recall,f1 = metrics_calc(tp_epoch,predict_len_epoch,gt_len_epoch)
        # print(precision,recall,f1)
    return running_loss / len(val_loader),precision,recall,f1
        
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

    return out_ind  # output index pairs: [batch, emo_clause, cause_clause]

def calculate_metrics(ground_truth, predictions, y_mask_b):
    TP = 0
    pred_len = 0
    gt_len = 0
    # ipdb.set_trace()
    predictions = predictions.tolist()
    num_sentences_per_doc = y_mask_b.sum(axis=1).tolist()
    
    for idx, pred in enumerate(predictions):
        col_idx = pred[0]
        
        # If number of predictions is larger than the number of sentences for a document, skip this prediction
        if pred[1] > num_sentences_per_doc[col_idx] or pred[2] > num_sentences_per_doc[col_idx]:
            continue
        
        pred_pairs = {tuple(pred[1:])}
        truth_pairs = {tuple(x) for x in ground_truth[col_idx]}
        # print("doc id: ", col_idx, "num_sent: ", num_sentences_per_doc[col_idx],  "pred_pairs: ", pred_pairs, "truth_pairs: ", truth_pairs)
        TP += len(pred_pairs.intersection(truth_pairs))
        pred_len += len(pred_pairs)
        gt_len += len(truth_pairs)
        
    return TP, pred_len, gt_len


# def check_accuracy_batch(doc_couples_b,res):
#     # print(doc_couples_b)
#     # print(res)
#     # print(res.shape)
#     # tt = (res[:,0]==0)
#     # print(tt.shape)
#     # batch_ind = res[:,0]
#     # pairs = res[:,1:]
#     # print(pairs)
#     # pairs = pairs + 1      # doc_couples_b starts from 1, while res starts from 0; don't use inplace plus (+=)
#     # print(pairs)
#     # ipdb.set_trace()
#     tp = 0
#     predict_len = 1
#     gt_len = 0
#     for i in range(config.batch_size):
#         target_span = (res[:,0]==i).nonzero()
#         # print(target_span.shape)
#         # target_span = (res[:,0]==0).nonzero()
#         # print(res)
#         # target_span = (res[:,0]==1).nonzero()
#         # print(res)
#         # target_span = (res[:,0]==2).nonzero()
#         # print(res)
#         # ipdb.set_trace()
#         if target_span.shape[0]==0:
#             tp += 0
#             predict_len += 0
#             gt_len += len(doc_couples_b[i])
#         else:
#             # print(target_span[-1][0].item())
#             # print(res)
#             # print(res.shape)
#             # ipdb.set_trace()
#             pairs = res[(target_span[0][0].item()):(target_span[-1][0].item()+1),1:]
#             pairs = pairs + 1
#             # print(pairs)
#             # print(pairs.shape)
#             pairs = pairs.tolist()
#             # print(pairs)
#             # ipdb.set_trace()
#             for target_pair in doc_couples_b[i]:
#                 if (target_pair in pairs):
#                     tp += 1
#             predict_len += len(pairs)
#             gt_len += len(doc_couples_b[i])
#             # ipdb.set_trace()

#     # print(f'tp:{tp},predict_len:{predict_len},gt_len:{gt_len}')
#     # ipdb.set_trace()
        
#     return tp,predict_len,gt_len

def metrics_calc(tp_epoch, predict_len_epoch, gt_len_epoch):
    precision = tp_epoch/predict_len_epoch if predict_len_epoch>0 else 0
    recall = tp_epoch/gt_len_epoch if gt_len_epoch>0 else 0
    f1 = 2 * (precision * recall)/(precision + recall) if precision + recall > 0 else 0
    return precision,recall,f1

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
        print(f'============================Epoch {epoch}/{EPOCHS}============================')
        train_loss,precision_train,recall_train,f1_train = train_loop(configs, model, train_set)
        # Calculate average loss for the epoch
        val_loss,precision_val,recall_val,f1_val = eval_loop(configs, model, val_set)
        print(f'Training Loss: {train_loss:.4f}, Precision: {precision_train:.4f}, Recall: {recall_train:.4f}, F1: {f1_train:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Precision: {precision_val:.4f}, Recall: {recall_val:.4f}, F1: {f1_val:.4f}')
    
    # test loop
    test_loss,precision_test,recall_test,f1_test = eval_loop(configs, model, test_set)
    print(f'Test Loss: {test_loss:.4f}, Precision: {precision_test:.4f}, Recall: {recall_test:.4f}, F1: {f1_test:.4f}')

if __name__ == "__main__":

    main()

