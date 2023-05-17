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

def load_data(configs, fold_id=1):
        
    train_loader = build_train_data(configs, fold_id=fold_id)
    if configs.split == 'split20':
        val_loader = build_inference_data(configs, fold_id=fold_id, data_type='valid')
        
    test_loader = build_inference_data(configs, fold_id=fold_id, data_type='test')
    return train_loader, val_loader, test_loader

def loss_mask(y_mask_b):
    num_sentences_per_doc = y_mask_b.sum(axis=1).tolist()
    y_list_mask_single = [torch.nn.functional.pad(torch.ones(num_sentences_per_doc[i],2),\
                                                    (0,0,0,configs.max_doc_len-num_sentences_per_doc[i]),value=0)\
                            for i in range(configs.batch_size)]
    y_list_mask_single = (torch.stack(y_list_mask_single,dim=0)).to(device=device)
    scores_mask = [torch.nn.functional.pad(torch.ones(num_sentences_per_doc[i],num_sentences_per_doc[i]),\
                                                    (0,configs.max_doc_len-num_sentences_per_doc[i],0,configs.max_doc_len-num_sentences_per_doc[i]),value=0)\
                            for i in range(configs.batch_size)]
    scores_mask = (torch.stack(scores_mask,dim=0)).to(device=device)

    return y_list_mask_single,scores_mask


def train_loop(configs, model, train_loader,epoch):

    model.train()

    with torch.no_grad():
        num_steps_all = len(train_loader) // configs.gradient_accumulation_steps * configs.EPOCHS
        warmup_steps = int(num_steps_all * configs.warmup_proportion)
    
    optimizer = optim.AdamW(model.parameters(), lr=configs.learning_rate, weight_decay=configs.weight_decay) 
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps_all)
    
    running_loss = 0.0
    
    with torch.no_grad():
        tp_epoch,predict_len_epoch,gt_len_epoch = 0,0,0
        sliding_mask = slidingmask_gen(D=configs.max_doc_len, 
                                        W=configs.window_size, 
                                        batch_size=configs.batch_size, 
                                        device=device)

    for train_step, batch in enumerate(train_loader, 1):
        with torch.no_grad():
            doc_len_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, \
            bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b = batch
            # ipdb.set_trace()
            
            bert_token_b = bert_token_b.to(device=device)
            bert_segment_b = bert_segment_b.to(device=device)
            bert_masks_b = bert_masks_b.to(device=device)
            bert_clause_b = bert_clause_b.to(device=device)

            y_list_mask_single,scores_mask = loss_mask(y_mask_b)
            sent_mask = (y_list_mask_single,scores_mask)

            
        
        y_e_list, y_c_list, s_final, cml_scores, eml_scores = model(bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b, y_mask_b)

        loss_total,cml_out,eml_out = loss_calc(configs,
                                                y_e_list,
                                                y_c_list,
                                                doc_couples_b,
                                                cml_scores,
                                                eml_scores,
                                                sliding_mask,
                                                sent_mask,
                                                epoch,
                                                training=True,
                                                alter=False,
                                                sent_mask_flag=True)
        with torch.no_grad():
            res = inference(cml_out, eml_out, y_mask_b, mode='logic_or')
            # todo: calculate metrics
            tp,predict_len,gt_len = check_accuracy_batch(doc_couples_b,res,y_mask_b)
            tp_epoch += tp
            predict_len_epoch += predict_len
            gt_len_epoch += gt_len
        
        # move the zero grad here
        optimizer.zero_grad()
        # Backward pass
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            running_loss += loss_total.item()

    with torch.no_grad():
        # metrics
        print(f'OUTPUT >>>> tp_epoch:{tp_epoch},predict_len_epoch:{predict_len_epoch},gt_len_epoch:{gt_len_epoch}')
        precision, recall, f1 = metrics_calc(tp_epoch,predict_len_epoch,gt_len_epoch)
        # print(precision,recall,f1)

    return running_loss / len(train_loader),precision,recall,f1

def eval_loop(configs, model, val_loader,epoch):
    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        tp_epoch,predict_len_epoch,gt_len_epoch = 0,0,0

        sliding_mask = slidingmask_gen(D=configs.max_doc_len, 
                                                    W=configs.window_size, 
                                                    batch_size=configs.batch_size, 
                                                    device=device)

        for val_step, batch in enumerate(val_loader, 1):
            doc_len_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, \
            bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b = batch
            
            bert_token_b = bert_token_b.to(device=device)
            bert_segment_b = bert_segment_b.to(device=device)
            bert_masks_b = bert_masks_b.to(device=device)
            bert_clause_b = bert_clause_b.to(device=device)

            y_list_mask_single,scores_mask = loss_mask(y_mask_b)
            sent_mask = (y_list_mask_single,scores_mask)
            

            y_e_list, y_c_list, s_final, cml_scores, eml_scores = model(bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b, y_mask_b)
            
            loss_total,cml_out,eml_out = loss_calc(configs,
                                                   y_e_list,
                                                    y_c_list,
                                                    doc_couples_b,
                                                    cml_scores,
                                                    eml_scores,
                                                    sliding_mask,
                                                    sent_mask,
                                                    epoch,
                                                    training=False,
                                                    alter=False,
                                                    sent_mask_flag=True)
            running_loss += loss_total.item()
            res = inference(cml_out, eml_out, y_mask_b, mode='logic_or')
            # todo: calculate metrics
            tp,predict_len,gt_len = check_accuracy_batch(doc_couples_b,res,y_mask_b)
            tp_epoch += tp
            predict_len_epoch += predict_len
            gt_len_epoch += gt_len

        precision,recall,f1 = metrics_calc(tp_epoch,predict_len_epoch,gt_len_epoch)

    return running_loss / len(val_loader),precision,recall,f1

def top_k_values(tensor, k=5):
    # Get the top 20 values from each row
    top_values, _ = torch.topk(tensor, k=k, dim=2)

    # Check the dimensions
    assert top_values.shape == (tensor.shape[0], tensor.shape[1], k)

    # Get the top 20 values from each 50x20 matrix (from each row)
    top_values, _ = torch.topk(top_values, k=k, dim=1)

    # Check the dimensions
    assert top_values.shape == (tensor.shape[0], k, k)

    return top_values

def inference(cml_out, eml_out, y_mask_b,mode='avg',topk=False):
    eml_out_T = torch.permute(eml_out, (0,2,1))
    # todo: topk

    # cml_out = top_k_values(cml_out)
    # eml_out_T = top_k_values(eml_out_T)
    
    if mode == 'avg':
        out = ((cml_out + eml_out_T)/2)>0.5
        out_ind = out.nonzero()
    elif mode == 'logic_and':
        # ipdb.set_trace()
        cml_pair = cml_out>0.5
        eml_pair = eml_out_T>0.5
        out = torch.logical_and(cml_pair, eml_pair)
        out_ind = out.nonzero()
    elif mode == 'logic_or':
        cml_pair = cml_out>0.5
        eml_pair = eml_out_T>0.5
        out = torch.logical_or(cml_pair, eml_pair)
        out_ind = out.nonzero()

    # remove prediction out of the range of sentences
    out_ind = out_ind.tolist()
    out_ind_filtered = []
    
    num_sentences_per_doc = y_mask_b.sum(axis=1).tolist()
    for pred in out_ind:
        batch_idx = pred[0]
        if (pred[1] > num_sentences_per_doc[batch_idx]) or (pred[2] > num_sentences_per_doc[batch_idx]):
            continue
        out_ind_filtered.append(pred)

    out_ind_filtered = torch.tensor(out_ind_filtered)
    return out_ind_filtered  # output index pairs: [batch, emo_clause, cause_clause]

def check_accuracy_batch(doc_couples_b,res,y_mask_b):
    tp = 0
    predict_len = 1
    gt_len = 0
    for i in range(configs.batch_size):
        if res.shape[0] == 0:
            tp += 0
            predict_len += 0
            gt_len += len(doc_couples_b[i])
        else:
            target_span = (res[:,0]==i).nonzero()
            if target_span.shape[0]==0:
                tp += 0
                predict_len += 0
                gt_len += len(doc_couples_b[i])
            else:
                
                pairs = res[(target_span[0][0].item()):(target_span[-1][0].item()+1),1:]
                pairs = pairs + 1 # since the ground truth start from 
                pairs = pairs.tolist()

                # num_sentences_per_doc = y_mask_b.sum(axis=1).tolist()
                # pairs = [pred for pred in pairs if not ((pred[0] > num_sentences_per_doc[i]) or (pred[1] > num_sentences_per_doc[i]))]

                # print(f"doc{i}, num of sentences{num_sentences_per_doc[i]}, num of pred_pairs{len(pairs)}")
                # ipdb.set_trace()
                for target_pair in doc_couples_b[i]:
                    if (target_pair in pairs):
                        tp += 1
                predict_len += len(pairs) 
                gt_len += len(doc_couples_b[i])

    # print(f'tp:{tp},predict_len:{predict_len},gt_len:{gt_len}')
        
    return tp,predict_len,gt_len

def metrics_calc(tp_epoch, predict_len_epoch, gt_len_epoch):
    precision = tp_epoch/predict_len_epoch if predict_len_epoch>0 else 0
    recall = tp_epoch/gt_len_epoch if gt_len_epoch>0 else 0
    f1 = 2 * (precision * recall)/(precision + recall) if precision + recall > 0 else 0
    return precision,recall,f1

def main(configs): 
    # print(configs.EPOCHS)
    # ipdb.set_trace()

    # load data 
    
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
    
    folder_num = configs.end_fold - configs.start_fold
    # train loop 
    for epoch in range(0, EPOCHS):  # change epoch starting from 0
        print(f'============================Epoch {epoch+1}/{EPOCHS}============================')
        train_losses, val_losses = 0.0, 0.0
        precision_sum_train, recall_sum_train, f1_sum_train = 0, 0, 0
        precision_sum_val, recall_sum_val, f1_sum_val = 0, 0, 0
        for fold_id in range(configs.start_fold, configs.end_fold): 
            print(f'OUTPUT >>>> fold:{fold_id}')
            train_set, val_set, _ = load_data(configs, fold_id)
            train_loss,precision_train,recall_train,f1_train = train_loop(configs, model, train_set,epoch)
            # Calculate average loss for the epoch
            val_loss,precision_val,recall_val,f1_val = eval_loop(configs, model, val_set,epoch)
            
            train_losses += train_loss
            val_losses += val_loss
            
            precision_sum_train += precision_train
            recall_sum_train += recall_train
            f1_sum_train += f1_train
            
            precision_sum_val += precision_val
            recall_sum_val += recall_val
            f1_sum_val += f1_val
            
        print(f'OUTPUT >>>> Training Loss: {train_losses/folder_num:.4f}, Precision: {precision_sum_train/folder_num:.4f}, Recall: {recall_sum_train/folder_num:.4f}, F1: {f1_sum_train/folder_num:.4f}')
        print(f'OUTPUT >>>> Validation Loss: {val_losses/folder_num:.4f}, Precision: {precision_sum_val/folder_num:.4f}, Recall: {recall_sum_val/folder_num:.4f}, F1: {f1_sum_val/folder_num:.4f}')
    
    # test loop
    print(f'============================ Testing ============================')
    test_losses = 0.0
    precision_sum_test, recall_sum_test, f1_sum_test = 0, 0, 0
    for fold_id in range(configs.start_fold, configs.end_fold):
        _,_,test_set = load_data(configs, fold_id)
        test_loss,precision_test,recall_test,f1_test = eval_loop(configs, model, test_set,epoch)
        
        test_losses += test_loss
        precision_sum_test += precision_test
        recall_sum_test += recall_test
        f1_sum_test += f1_test
    print(f'OUTPUT >>>> Test Loss: {test_losses/folder_num:.4f}, Precision: {precision_sum_test/folder_num:.4f}, Recall: {recall_sum_test/folder_num:.4f}, F1: {f1_sum_test/folder_num:.4f}')

if __name__ == "__main__":
    # hyperparameter tuning
    configs = Config()
    # print(configs.__dict__)
    # ipdb.set_trace()
    # print(configs.EPOCHS)
    mod_para = {'EPOCHS':5}
    for key in mod_para:
        configs.__dict__[key] = mod_para[key]
    # print(configs.EPOCHS)
    # ipdb.set_trace()
    main(configs)

