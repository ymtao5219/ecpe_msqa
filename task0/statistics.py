import json
import os

def merge_JsonFiles(filename):
    result = list()
    for f1 in filename:
        with open(f1, 'r', encoding='utf-8') as infile:
            result.extend(json.load(infile))
    return result

def read_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as fr:
        js = json.load(fr)
    return js

def counting(data_list): # json object
    emotion_category = []
    distances = []
    num_pairs = []
    doc_ids = []
    for doc in data_list:
        doc_id = doc['doc_id']
        if doc_id  not in doc_ids:
            doc_len = doc['doc_len'] # calaulate number of documents with n pairs
            doc_pairs = doc['pairs']
            num_pairs.append(len(doc_pairs)) 

            for pair in doc_pairs: # calaulate distance of pairs
                distance = pair[1] - pair[0]
                distances.append(distance)

            doc_clauses = doc['clauses']
            for i in range(doc_len):
                clause = doc_clauses[i]
                clause_id = clause['clause_id']
                if clause['emotion_category'] != 'null':
                    emotion_category.append(clause['emotion_category']) # calaulate different emotions
            doc_ids.append(doc_id)
    return distances, num_pairs, emotion_category, doc_ids
    
