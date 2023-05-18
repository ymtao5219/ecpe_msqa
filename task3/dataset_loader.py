import sys
sys.path.append('..')
from os.path import join
import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from config import *
import json

torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)
torch.backends.cudnn.deterministic = True


# def build_train_data(configs, fold_id, shuffle=True):
#     train_dataset = MyQADataset(configs, fold_id, data_type='train')
#     train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
#                                                shuffle=shuffle, collate_fn=bert_batch_preprocessing)
#     return train_loader


# def build_inference_data(configs, fold_id, data_type):
#     dataset = MyQADataset(configs, fold_id, data_type)
#     data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=configs.batch_size,
#                                               shuffle=False, collate_fn=bert_batch_preprocessing)
#     return data_loader

class MyQADataset(Dataset):
    def __init__(self, configs, fold_id, question, data_type, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.split = configs.split
        self.train_file = join(data_dir, self.split, TRAIN_FILE % fold_id)
        self.valid_file = join(data_dir, self.split, VALID_FILE % fold_id)
        self.test_file = join(data_dir, self.split, TEST_FILE % fold_id)
        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.model_name)
        self.data_type = data_type
        self.question = question

        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'valid':
            data_file = self.valid_file
        elif data_type == 'test':
            data_file = self.test_file

        data_list = read_json(data_file)
        self.files = []
        for doc in data_list:
            doc_len = doc['doc_len']
            doc_clauses = doc['clauses']
            doc_str = ''
            for i in range(doc_len):
                clause = doc_clauses[i]
                clause_id = clause['clause_id']
                assert int(clause_id) == i + 1
                clause_str = clause_id + clause['clause']
                doc_str += clause_str
            self.files.append(clause_str)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        text = self.files[idx]
        # print(text)
        input_ids = self.bert_tokenizer.encode(self.question, text)
        return input_ids
    
def read_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as fr:
        js = json.load(fr)
    return js