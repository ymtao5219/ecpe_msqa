import pandas as pd
import os

def load_data(files, path):
    result = []
    for f in files:
        fpath = os.path.join(path, f)
        df = pd.read_json(fpath)
        result.append(df)
            # result.append(pd.DataFrame(data.read()))
    return result


def parse_data(data, k=-1):
    '''k passages'''
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
        result.append({'content': content, 'pair': e[1].pairs[0], 'emotion_category': emotion_c, 'emotion_token': emotion_t})
    return result