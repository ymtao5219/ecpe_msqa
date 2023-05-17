import torch
import torch.nn as nn
from transformers import BertModel
from config import DEVICE
import torch.nn.functional as F
from config import *
# config = Config()

import ipdb

#####################################################################################################
# Layers/Blocks
#####################################################################################################
class PretrainedBERT(nn.Module):
    def __init__(self, model_name, freeze=True):
        super(PretrainedBERT, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.linear = nn.Linear(768, 768)
        self.activation = nn.ReLU()
        
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
        doc_sents_h = self.activation(self.linear(doc_sents_h))
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
    
# class WordAttention(nn.Module):
#     def __init__(self, hidden_size):
#         super(WordAttention, self).__init__()
#         self.hidden_size = hidden_size
#         self.attention = nn.Linear(2 * hidden_size, 1)
#         self.activation = nn.Tanh()
        
#     def forward(self, bilstm_output):
#         # bilstm_output shape: (batch_size, max_seq_length, 2 * hidden_size)
        
#         attention_scores = self.attention(bilstm_output)
#         # attention_scores shape: (batch_size, max_seq_length, 1)

#         attention_weights = torch.softmax(attention_scores, dim=1)
#         # attention_weights shape: (batch_size, max_seq_length, 1)

#         weighted_bilstm_output = attention_weights * bilstm_output
#         # weighted_bilstm_output shape: (batch_size, max_seq_length, 2 * hidden_size)

#         weighted_bilstm_output = self.activation(weighted_bilstm_output)
        
#         return weighted_bilstm_output
    
class WordAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=5):
        super(WordAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.query_projection = nn.Linear(2 * hidden_size, hidden_size)
        self.key_projection = nn.Linear(2 * hidden_size, hidden_size)
        self.value_projection = nn.Linear(2 * hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, 2 * hidden_size)
        self.activation = nn.ReLU()

    def forward(self, inputs):
        batch_size, max_seq_length, _ = inputs.size()

        # Project the inputs to query, key, and value tensors
        queries = self.activation(self.query_projection(inputs))
        keys = self.activation(self.key_projection(inputs))
        values = self.activation(self.value_projection(inputs))

        # Reshape the projected tensors to split into multiple heads
        queries = queries.view(batch_size, max_seq_length, self.num_heads, self.head_size)
        keys = keys.view(batch_size, max_seq_length, self.num_heads, self.head_size)
        values = values.view(batch_size, max_seq_length, self.num_heads, self.head_size)

        # Transpose to move the head dimension to the batch dimension
        queries = queries.transpose(1, 2)  # [batch_size, num_heads, max_seq_length, head_size]
        keys = keys.transpose(1, 2)  # [batch_size, num_heads, max_seq_length, head_size]
        values = values.transpose(1, 2)  # [batch_size, num_heads, max_seq_length, head_size]

        # Compute scaled dot-product attention for each head
        attention_scores = torch.matmul(queries, keys.transpose(-1, -2)) / (self.head_size ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)

        # Apply attention to values
        context = torch.matmul(attention_probs, values)

        # Transpose and reshape the context tensor
        context = context.transpose(1, 2)  # [batch_size, max_seq_length, num_heads, head_size]
        context = context.contiguous().view(batch_size, max_seq_length, self.hidden_size)

        # Project the context tensor to obtain the output
        outputs = self.activation(self.output_projection(context))

        return outputs

class ISMLBlock(nn.Module):
    def __init__(self, N, D, hidden_size):
        super(ISMLBlock, self).__init__()
        self.N = N
        self.D = D
        self.hidden_size = hidden_size
        
        self.bilstm_e_list = nn.ModuleList([])
        self.bilstm_c_list = nn.ModuleList([])
        self.fc_e_list = nn.ModuleList([])
        self.fc_c_list = nn.ModuleList([])
        
        self.activation = nn.LeakyReLU() #nn.ReLU()
        
        for n in range(N):
            self.bilstm_e_list.append(nn.LSTM(input_size= hidden_size*2+4*n, hidden_size= hidden_size,\
                               num_layers= 4, batch_first=True,bidirectional=True))

            self.bilstm_c_list.append(nn.LSTM(input_size= hidden_size*2+4*n, hidden_size= hidden_size,\
                               num_layers= 4, batch_first=True,bidirectional=True))

            self.fc_e_list.append(nn.Linear(hidden_size*2,2))
            self.fc_c_list.append(nn.Linear(hidden_size*2,2))

        self.fc_cml = nn.Linear(hidden_size*2,D)
        self.fc_eml = nn.Linear(hidden_size*2,D)

    def forward(self, s1, D,y_mask):
        self.y_e_list = []
        self.y_c_list = []

        
        
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

        # with torch.no_grad():
        #     mask = input_padding(y_mask,D)
        
        for n in range(self.N):
            # print(self.bilstm_e_list[n](s_tmp))
            e_lstm_out,_ = self.bilstm_e_list[n](s_tmp)
            # print(e_lstm_out.get_device())
            # ipdb.set_trace()
            y_e = nn.functional.softmax(self.fc_e_list[n](e_lstm_out),dim=2)
            # mask_expanded = mask.expand_as(y_e)
            # y_e = y_e * mask_expanded
            self.y_e_list.append(y_e)

            c_lstm_out,_ = self.bilstm_c_list[n](s_tmp)
            y_c = nn.functional.softmax(self.fc_c_list[n](c_lstm_out),dim=2)
            # mask_expanded = mask.expand_as(y_c)
            # y_c = y_c * mask_expanded
            self.y_c_list.append(y_c)

            s_tmp = torch.cat((s_tmp,y_e,y_c),dim=2)

        e_lstm_out = self.activation(e_lstm_out)
        c_lstm_out = self.activation(c_lstm_out)
        # print(e_lstm_out.get_device())
        # ipdb.set_trace()
        
        
        # e_lstm_out = e_lstm_out * mask
        # c_lstm_out = c_lstm_out * mask
        
        # mask = mask * mask.transpose(-1, -2) 
        # cml_scores = self.fc_cml(e_lstm_out) * mask
        # eml_scores = self.fc_eml(c_lstm_out) * mask
        cml_scores = self.fc_cml(e_lstm_out)
        eml_scores = self.fc_eml(c_lstm_out)

        # print(cml_scores.get_device())
        # ipdb.set_trace()
        
        return self.y_e_list,self.y_c_list,s_tmp,cml_scores,eml_scores

#####################################################################################################
# Full Model
#####################################################################################################
class Network(nn.Module):
    def __init__(self, model_name="bert-base-chinese", 
                       max_sen_len=30, 
                       max_doc_len=75, 
                       max_doc_len_bert=350,
                       model_iter_num=3, 
                       window_size=3, 
                       n_hidden=100, 
                       n_class=2):
        
        super(Network, self).__init__()

        self.max_doc_len = max_doc_len
        self.n_hidden = n_hidden
        self.model_iter_num = model_iter_num
        
        self.bert = PretrainedBERT(model_name)
        self.biLSTM = BiLSTM(768, n_hidden, 4)
        self.word_attention = WordAttention(n_hidden)
        self.isml_block = ISMLBlock(model_iter_num, max_doc_len, n_hidden)

    def forward(self, bert_token_b, bert_segment_b, bert_masks_b,
                bert_clause_b, y_mask):
        
        y_mask = torch.tensor(y_mask).to(bert_clause_b.device).unsqueeze(-1)
        x = self.bert(bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b)
        # x = x * y_mask 
        x = self.biLSTM(x)
        # x = x * y_mask
        x = self.word_attention(x)
        # x = x * y_mask
        x = input_padding(x,self.max_doc_len)
        x = self.isml_block(x, self.max_doc_len, y_mask)
        return x

#####################################################################################################
# helper functions
#####################################################################################################
# D = config.max_doc_len
def input_padding(s1,target):  # target --> max doc length
    s1 = torch.nn.functional.pad(s1,(0,0,0,target-s1.shape[1]),value=0)
    return s1

def slidingmask_gen(D, W, batch_size, device):
    slidingmask = torch.ones(D,D)  
    slidingmask = torch.triu(slidingmask,diagonal=-W)  
    slidingmask = torch.tril(slidingmask,diagonal=W) 
    slidingmask = slidingmask.repeat(batch_size,1,1)
    slidingmask = slidingmask.to(device=device)
    # print(slidingmask)
    return slidingmask