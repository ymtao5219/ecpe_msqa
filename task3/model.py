import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel
from config import DEVICE
import torch.nn.functional as F

from torchcrf import CRF

import ipdb
class PretrainedBERT(nn.Module):
    def __init__(self, model_name, freeze=True):
        super(PretrainedBERT, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.roberta = RobertaModel.from_pretrained(model_name)
        
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
    

class Pooler(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(Pooler, self).__init__()
        self.num_labels = num_labels
        self.bilstm = nn.LSTM(hidden_size, hidden_size // 2, num_layers=1, bidirectional=True)
        self.crf = CRF(num_labels, batch_first=True)
        self.hidden2label = nn.Linear(hidden_size, num_labels)

    def forward(self, last_hidden_states, labels=None):
        # Shape of outputs: [batch_size, sequence_length, hidden_size]
        outputs, _ = self.bilstm(last_hidden_states)
        
        # Shape of emissions: [batch_size, sequence_length, num_labels]
        emissions = self.hidden2label(outputs)
        
        if labels is not None:
            loss = -self.crf(emissions, labels, reduction='mean')
            return loss
        else:
            best_paths = self.crf.decode(emissions)
            return best_paths
