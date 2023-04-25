import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, bidirectional=True):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, seq_lengths):
        packed_x = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return out

class WordAttention(nn.Module):
    def __init__(self, input_size):
        super(WordAttention, self).__init__()
        self.att = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, 1)
        )

    def forward(self, x, seq_lengths):
        att_weights = self.att(x)
        mask = torch.arange(x.size(1)).unsqueeze(0) >= seq_lengths.unsqueeze(1)
        att_weights.masked_fill_(mask.unsqueeze(-1).to(att_weights.device), -float('inf'))
        att_weights = torch.softmax(att_weights, dim=1)
        return (x * att_weights).sum(dim=1)

class EmoCausePrediction(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EmoCausePrediction, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        return torch.softmax(self.fc(x), dim=-1)

class ISMLModel(nn.Module):
    def __init__(self, model_name, n_hidden, max_doc_len, max_sen_len, n_class, model_type='Inter-EC', model_iter_num=3):
        super(ISMLModel, self).__init__()

        self.max_doc_len = max_doc_len
        self.n_hidden = n_hidden
        self.model_type = model_type
        self.model_iter_num = model_iter_num

        # Load pre-trained BERT model
        self.bert_config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name, config=self.bert_config)

        self.word_attention = WordAttention(2 * n_hidden)
        self.sentence_encoder = BiLSTM(2 * n_hidden, n_hidden)
        self.emo_cause_predictor = EmoCausePrediction(2 * n_hidden, n_class)

    def forward(self, x_bert, x_mask_bert, x_type_bert, s_idx_bert, x, sen_len, doc_len, is_training):
        def get_bert_s(x_bert, x_mask_bert, x_type_bert, s_idx_bert, is_training, feature_mask):
            with torch.no_grad():
                bert_output = self.bert(x_bert, attention_mask=x_mask_bert, token_type_ids=x_type_bert)
            s_bert = bert_output.last_hidden_state

            # [-1, max_doc_len_bert, n_hidden]
            batch_size, n_hidden = s_bert.size(0), s_bert.size(-1)
            index = torch.arange(0, batch_size).unsqueeze(1) * max_doc_len + s_idx_bert
            index = index.view(-1)
            s_bert = s_bert.view(-1, n_hidden).index_select(0, index)  # batch_size * n_hidden
            s_bert = s_bert.view(-1, self.max_doc_len, n_hidden)

            # [-1, max_doc_len, n_hidden]
            s_bert = s_bert * feature_mask  # independent clause representation from BERT
            s_bert = torch.nn.functional.linear(s_bert, weight=torch.empty(n_hidden, 2 * self.n_hidden), bias=torch.empty(2 * self.n_hidden))

            # shape: [-1, max_doc_len, 2 * n_hidden]
            return s_bert

        def get_s(inputs, sen_len, name):
            inputs = self.sentence_encoder(inputs, sen_len)
            s = self.word_attention(inputs, sen_len)
            s = s.view(-1, self.max_doc_len, 2 * self.n_hidden)
            return s

        def emo_cause_prediction(s_ec, is_training, name):
            pred_ec = self.emo_cause_predictor(s_ec)
            pred_ec = pred_ec.view(-1, self.max_doc_len, n_class)
            return pred_ec

        def getmask(lengths, max_len, shape):
            mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.float().view(shape).to(lengths.device)
            return mask

        cause_list, emo_list, reg = [], [], 0
        feature_mask = getmask(doc_len, self.max_doc_len, [-1, self.max_doc_len, 1])

        if self.model_type in ['Inter-CE', 'Inter-EC']:
            s_ec = get_s(x, sen_len, name='word_encode_emotion')
            s_emo = self.sentence_encoder(s_ec, doc_len)
            pred_emo = emo_cause_prediction(s_emo, is_training, name='emotion')

            s_ec = get_s(x, sen_len, name='word_encode_cause')
            s_ec = torch.cat([s_ec, pred_emo], dim=2) * feature_mask
            s_cause = self.sentence_encoder(s_ec, doc_len)
            pred_cause = emo_cause_prediction(s_cause, is_training, name='cause')

            emo_list.append(pred_emo)
            cause_list.append(pred_cause)

            if self.model_type == 'Inter-CE':
                cause_list, emo_list, s_cause, s_emo = emo_list, cause_list, s_emo, s_cause
        elif self.model_type == 'ISML':
            for i in range(self.model_iter_num):
                s_ec = get_s(x, sen_len, name=f'word_encode_emotion_{i}')
                s_ec = torch.cat([s_ec] + cause_list + emo_list, dim=2) * feature_mask
                s_emo = self.sentence_encoder(s_ec, doc_len)
                pred_emo = emo_cause_prediction(s_emo, is_training, name=f'emotion_{i}')

                s_ec = get_s(x, sen_len, name=f'word_encode_cause_{i}')
                s_ec = torch.cat([s_ec] + cause_list + emo_list, dim=2) * feature_mask
                s_cause = self.sentence_encoder(s_ec, doc_len)
                pred_cause = emo_cause_prediction(s_cause, is_training, name=f'cause_{i}')

                emo_list.append(pred_emo)
                cause_list.append(pred_cause)
        else:
            s_bert = get_bert_s(x_bert, x_mask_bert, x_type_bert, s_idx_bert, is_training, feature_mask)
            
            s_emo = self.sentence_encoder(s_bert, doc_len)
            pred_emo = emo_cause_prediction(s_emo, is_training, name='emotion')

            s_cause = self.sentence_encoder(s_bert, doc_len)
            pred_cause = emo_cause_prediction(s_cause, is_training, name='cause')

            emo_list.append(pred_emo)
            cause_list.append(pred_cause)

        return emo_list, cause_list, s_emo, s_cause

