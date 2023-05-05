import json

def read_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as fr:
        js = json.load(fr)
    return js

# todo: evaluate function 
# f1, precision, recall, accuracy