import pandas as pd
import os

def read_file_under_paths(path, end='.json'):
    '''read data file name into list'''
    text_files = [f for f in os.listdir(path)]
    result = [f for f in text_files if f.endswith(end)]
    return result

def num_seq(x):
    '''helper used to sort'''
    offset = 4
    try:
       return int(x[offset:offset+2])
    except:
       return int(x[offset])

def load_data(file_names, path):
    '''load the data from file name to pandas df'''
    result = []
    for f in sorted(file_names, key=num_seq):
        fpath = os.path.join(path, f)
        df = pd.read_json(fpath)
        result.append(df)
            # result.append(pd.DataFrame(data.read()))
    return result


def parse_data(data, k=None):
    '''parse k passages'''
    if k:
        data = data[:k]
    # result = [{'pair': e[1].pairs[0], 'clauses': e[1].clauses} for e in data.iterrows()]
    result = []
    for e in data.iterrows():
        content = ''
        emotion_c = []
        emotion_t = []
        index = 1
        for k in e[1].clauses:
            content += f'{index} ' + k['clause'] + '\n'
            if k['emotion_category'] != 'null':
                emotion_c.append(k['emotion_category'])
            if k['emotion_token'] != 'null':
                emotion_t.append(k['emotion_token'])  
            index += 1
        result.append({'content': content, 'pair': e[1].pairs, 'emotion_category': emotion_c, 'emotion_token': emotion_t})
    return result
