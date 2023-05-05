import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_SEED = 129
DATA_DIR = '../../data_json'
TRAIN_FILE = 'fold%s_train.json'
VALID_FILE = 'fold%s_valid.json'
TEST_FILE  = 'fold%s_test.json'

class Config:
    def __init__(self):
        ## input struct ##
        self.model_name = "bert-base-chinese"
        self.bert_cache_path = 'bert_base_chinese/'
        self.max_sen_len = 30
        self.max_doc_len = 75
        self.max_doc_len_bert = 350
        ## model struct ##
        self.model_iter_num = 1
        self.model_type = 'Inter-EC'
        self.window_size = 3
        self.n_hidden = 100
        self.n_class = 2
        ## For Training ##
        self.start_fold = 1
        self.end_fold = 11
        self.split = 'split20'
        self.batch_size = 8
        self.learning_rate = 2e-5
        self.keep_prob1 = 0.5
        self.keep_prob2 = 1.0
        self.l2_reg = 1e-5
        self.emo = 1.0
        self.cause = 1.0
        self.pair = 1.0
        self.threshold = 0.5
        self.training_iter = 20
        self.log_file_name = ''

