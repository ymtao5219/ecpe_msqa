import torch
import torch.nn as nn
from transformers import BertModel
from config import DEVICE
import torch.nn.functional as F

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
    def __init__(self):
        super(ISMLBlock, self).__init__()
        pass 
    
    def forward(self):
        pass
    
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
