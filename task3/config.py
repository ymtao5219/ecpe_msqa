import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_SEED = 129
DATA_DIR = '../data_json'
TRAIN_FILE = 'fold%s_train.json'
VALID_FILE = 'fold%s_valid.json'
TEST_FILE  = 'fold%s_test.json'

class Config:
    def __init__(self):
        ## input struct ##
        self.model_name = "bert-base-chinese"
        ## For Training ##
        self.start_fold = 1
        self.end_fold = 11
        self.split = 'split20'
        self.batch_size = 1
        self.learning_rate = 2e-5
        self.log_file_name = ''

