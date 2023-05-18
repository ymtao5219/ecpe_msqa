import sys
sys.path.append('..')
from transformers import BertTokenizer
from config import *
import torch
from transformers import BertForQuestionAnswering
from config import *
from utils import *
from dataset_loader import MyQADataset
import torch.utils.data as Data

def val(epoch, val_loader, show=True):
    model.eval()
    # val_loss = 0
    with torch.no_grad():
        for batch_idx, input_ids in enumerate(val_loader):
            tokenizer = BertTokenizer.from_pretrained(configs.model_name)
            tokens = tokenizer.convert_ids_to_tokens(input_ids) # miss ground truth position to get loss
            output = model(torch.tensor([input_ids]))
            answer_start = torch.argmax(output.start_logits)
            answer_end = torch.argmax(output.end_logits)
            answer = " ".join(tokens[answer_start:answer_end+1])
            print(answer) 
    return # val_loss/(batch_idx+1.0)
    
if __name__ == '__main__':
    # Try with one example
    configs = Config()
    model = BertForQuestionAnswering.from_pretrained(configs.model_name)
    tokenizer = BertTokenizer.from_pretrained(configs.model_name)
    question = '文档的每一行字代表一个字句，每句开头的数字代表该子句的编号。我们将包含情绪表达的子句称为情绪子句，将导致情绪发生的子句称为原因子句。基于以上内容，将具有因果关系的子句匹配成‘（情绪子句编号，原因子句编号）的形式。'
    val_set=MyQADataset(configs, fold_id=1, question=question, data_type='valid', data_dir=DATA_DIR)
    val_loader = Data.DataLoader(val_set, batch_size=configs.batch_size, shuffle=False)
    print('loading datasets is done')
    val(0, val_loader, show=True)



    









            


