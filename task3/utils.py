import json


def eval_func(doc_couples_all, doc_couples_pred_all):
    pass 

def read_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as fr:
        js = json.load(fr)
    return js