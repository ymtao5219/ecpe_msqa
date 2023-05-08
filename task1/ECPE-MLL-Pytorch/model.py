import torch
import torch.nn as nn
from transformers import BertModel
from config import DEVICE
import torch.nn.functional as F
from config import *
config = Config()

import ipdb
class PretrainedBERT(nn.Module):
    def __init__(self, model_name, freeze=True):
        super(PretrainedBERT, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)

        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def forward(self, bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b):
        bert_output = self.bert(input_ids=bert_token_b,
                attention_mask=bert_masks_b,
                token_type_ids=bert_segment_b)
        hidden_state = bert_output[0]
        # ipdb.set_trace()
        dummy = bert_clause_b.unsqueeze(2).expand(bert_clause_b.size(0), bert_clause_b.size(1), hidden_state.size(2))
        doc_sents_h = hidden_state.gather(1, dummy)
        return doc_sents_h
    
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, 
                              batch_first=batch_first, bidirectional=True)

    def forward(self, bert_output):

        # Pass the BERT output through the BiLSTM
        output, _ = self.bilstm(bert_output)
        # output shape: (batch_size, max_seq_length, 2*hidden_size)
        return output
class WordAttention(nn.Module):
    def __init__(self, hidden_size):
        super(WordAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(2 * hidden_size, 1)

    def forward(self, bilstm_output):
        # bilstm_output shape: (batch_size, max_seq_length, 2 * hidden_size)
        
        attention_scores = self.attention(bilstm_output)
        # attention_scores shape: (batch_size, max_seq_length, 1)

        attention_weights = F.softmax(attention_scores, dim=1)
        # attention_weights shape: (batch_size, max_seq_length, 1)

        weighted_bilstm_output = attention_weights * bilstm_output
        # weighted_bilstm_output shape: (batch_size, max_seq_length, 2 * hidden_size)

        return weighted_bilstm_output
class ISMLBlock(nn.Module):
    def __init__(self, N, D, hidden_size):
        super(ISMLBlock, self).__init__()
        self.N = N
        self.D = D
        self.hidden_size = hidden_size
        
        self.bilstm_e_list = []
        self.bilstm_c_list = []
        self.fc_e_list = []
        self.fc_c_list = []
        
        for n in range(N):
            bilstm_e = nn.LSTM(input_size= hidden_size*2+4*n, hidden_size= hidden_size,\
                               num_layers= 1, batch_first=True,bidirectional=True)
            self.bilstm_e_list.append(bilstm_e)

            bilstm_c = nn.LSTM(input_size= hidden_size*2+4*n, hidden_size= hidden_size,\
                               num_layers= 1, batch_first=True,bidirectional=True)
            self.bilstm_c_list.append(bilstm_c)

            fc_e = nn.Linear(hidden_size*2,2)
            # nn.init.kaiming_normal_(fc_e.weight)
            self.fc_e_list.append(fc_e)

            fc_c = nn.Linear(hidden_size*2,2)
            # nn.init.kaiming_normal_(fc_c.weight)
            self.fc_c_list.append(fc_c)

        self.fc_cml = nn.Linear(hidden_size*2,D)
        self.fc_eml = nn.Linear(hidden_size*2,D)


    def forward(self, s1):
        # scores = None
        self.y_e_list = []
        self.y_c_list = []
        # s_tmp = s1
        
        s_tmp = s1.to(next(self.parameters()).device)  # move s1 to the same device as the module parameters
        
        # move all module parameters to the same device as s1
        for bilstm_e in self.bilstm_e_list:
            bilstm_e.to(s1.device)
        for fc_e in self.fc_e_list:
            fc_e.to(s1.device)
        for bilstm_c in self.bilstm_c_list:
            bilstm_c.to(s1.device)
        for fc_c in self.fc_c_list:
            fc_c.to(s1.device)
        self.fc_cml.to(s1.device)
        self.fc_eml.to(s1.device)


        for n in range(self.N):
            # print(self.bilstm_e_list[n](s_tmp))
            e_lstm_out,_ = self.bilstm_e_list[n](s_tmp)
            y_e = nn.functional.softmax(self.fc_e_list[n](e_lstm_out),dim=2)
            self.y_e_list.append(y_e)

            c_lstm_out,_ = self.bilstm_c_list[n](s_tmp)
            y_c = nn.functional.softmax(self.fc_c_list[n](c_lstm_out),dim=2)
            self.y_c_list.append(y_c)

            s_tmp = torch.cat((s_tmp,y_e,y_c),dim=2)
            
            # print('s_tmp shape',s_tmp.shape)

        cml_scores = self.fc_cml(e_lstm_out)
        eml_scores = self.fc_eml(c_lstm_out)


        return self.y_e_list,self.y_c_list,s_tmp,cml_scores,eml_scores


class Network(nn.Module):
    def __init__(self, model_name="bert-base-chinese", max_sen_len=30, max_doc_len=75, max_doc_len_bert=350,
                 model_iter_num=1, model_type='Inter-EC', window_size=3, n_hidden=100, n_class=2):
        super(Network, self).__init__()

        self.max_doc_len = max_doc_len
        self.n_hidden = n_hidden
        self.model_iter_num = model_iter_num
        
        self.bert = PretrainedBERT(model_name)
        self.biLSTM = BiLSTM(768, n_hidden, 1)
        self.word_attention = WordAttention(n_hidden)

    def forward(self, bert_token_b, bert_segment_b, bert_masks_b,
                bert_clause_b):
        
        x = self.bert(bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b)
        x = self.biLSTM(x)
        x = self.word_attention(x)
        # ipdb.set_trace()
        return x

#####################################################################################################
# helper functions
#####################################################################################################
D = config.max_doc_len
def input_padding(s1,len_target=D):  # D = 75 --> max doc length
    s1 = torch.nn.functional.pad(s1,(0,0,0,D-s1.shape[1]),value=0)
    return s1

def slidingmask_gen(D, W, batch_size, device):
    slidingmask = torch.ones(D,D)  
    slidingmask = torch.triu(slidingmask,diagonal=-W)  
    slidingmask = torch.tril(slidingmask,diagonal=W) 
    slidingmask = slidingmask.repeat(batch_size,1,1)
    slidingmask.to(device=device)
    # print(slidingmask)
    return slidingmask