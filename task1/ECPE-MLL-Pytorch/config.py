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
        self.max_sen_len = 30 # 30 # max number of tokens per clause
        self.max_doc_len = 50 # 75 # max number of clauses per document
        self.max_doc_len_bert = 350 # max number of tokens per document for Bert Model
        ## model struct ##
        self.model_iter_num = 6 # iter num of ISML
        self.window_size = 3
        self.n_hidden = 100
        self.n_class = 2
        # self.start_fold = 1
        # self.end_fold = 3      # 11 max --> fold 10
        self.split = 'split20'

        self.batch_size = 36
        self.learning_rate = 0.001 # 0.005 in the paper
        self.keep_prob1 = 0.5
        self.keep_prob2 = 1.0
        self.weight_decay = 1e-5   #1e-5 in the paper
        self.lamb_1 = 1.0 
        self.lamb_2 = 1.0
        self.lamb_3 = 1.0
        self.threshold = 0.5
        self.log_file_name = ''
        
        self.adj_param = 75
        self.gradient_accumulation_steps = 2
        self.warmup_proportion = 0.1

        if self.split == 'split10':
            self.start_fold = 1
            self.end_fold = 11      # 11 max --> 10-fold
            self.EPOCHS = 20
        elif self.split == 'split20':
            self.start_fold = 1
            self.end_fold = 5      # 21 max --> 20-fold
            self.EPOCHS = 10
        else:
            print('Unknown data split.')
            exit()

    # def para_mod(self,**mod_para):
    #     for key in mod_para:
    #         if key == 'EPOCHS':
    #             self.EPOCHS = mod_para[key]