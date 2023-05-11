
import torch 
from config import *
import ipdb

batch_size = 8
device = DEVICE

configs = Config()
N = configs.model_iter_num              # the repeating block number of ISML, should be >= 1
D = configs.max_doc_len
hidden_size = configs.n_hidden
batch_size = configs.batch_size
lamb_1 = configs.lamb_1
lamb_2 = configs.lamb_2
lamb_3 = configs.lamb_3
adj_param = configs.adj_param

def labelTransform(doc_couples_b):
    # batch_size = len(doc_couples_b)
    y_e_isml = torch.zeros(batch_size, D, 2)
    y_c_isml = torch.zeros(batch_size, D, 2)
    
    for i in range(batch_size):
        for emo,cau in doc_couples_b[i]:
            # print(emo,cau)
            y_e_isml[i][emo-1][0] = 1 # the True prob is col 0 and the False prob is col 1
            y_c_isml[i][cau-1][0] = 1
    # ipdb.set_trace()
    y_e_isml.to(device=device)
    y_c_isml.to(device=device)

    y_cml_pairs = torch.zeros(batch_size,D,D)
    y_eml_pairs = torch.zeros(batch_size,D,D)
    for i in range(batch_size):
        for emo,cau in doc_couples_b[i]:
            y_cml_pairs[i][emo-1][cau-1] = 1
            y_eml_pairs[i][cau-1][emo-1] = 1

    y_cml_pairs.to(device=device)
    y_eml_pairs.to(device=device)

    return y_e_isml,y_c_isml,y_cml_pairs,y_eml_pairs

def loss_calc(y_e_list,y_c_list,doc_couples_b,cml_scores,eml_scores,slidingmask):
    y_e_isml,y_c_isml,y_cml_pairs,y_eml_pairs = labelTransform(doc_couples_b)

    loss_isml = 0
    # ipdb.set_trace()
    for n in range(N):  # can accelerate by n times with full vectorization
        # print(y_e_isml.shape,y_e_list[n].shape)
        loss_isml += -torch.sum(torch.mul(y_e_isml,torch.log(y_e_list[n])))\
                        -torch.sum(torch.mul(y_c_isml,torch.log(y_c_list[n])))
        
    
    cml_out_beforemask = torch.div(1,1+torch.exp(cml_scores))
    eml_out_beforemask = torch.div(1,1+torch.exp(eml_scores))
    loss_cmll = -torch.sum(torch.mul(slidingmask,(torch.mul(y_cml_pairs,torch.log(cml_out_beforemask))\
                                    +torch.mul(1-y_cml_pairs,torch.log(1-cml_out_beforemask)) / adj_param )))
    loss_emll = -torch.sum(torch.mul(slidingmask,(torch.mul(y_eml_pairs,torch.log(eml_out_beforemask))\
                                    +torch.mul(1-y_eml_pairs,torch.log(1-eml_out_beforemask)) / adj_param )))
    
    with torch.no_grad():
        cml_out = torch.mul(slidingmask, cml_out_beforemask)
        eml_out = torch.mul(slidingmask, eml_out_beforemask)
    
    loss_total = lamb_1 * loss_isml + lamb_2 * loss_cmll + lamb_3 * loss_emll
    print(f'loss_total:{loss_total:.4f}, loss_isml:{loss_isml:.4f}, loss_cmll:{loss_cmll:.4f},loss_emll:{loss_emll:.4f}')
        
    return loss_total,cml_out,eml_out
