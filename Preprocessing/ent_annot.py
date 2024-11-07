#!/usr/bin/env python
# coding: utf-8

import json
import pandas as pd
import numpy as np
import itertools
from pathlib import Path
import stanza

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Download and configure Stanza model
stanza.download('en')
user_path = Path.home()
config = {
    'dir': f'{user_path}/stanza_resources',
    'processors': 'tokenize,pos,ner,depparse,lemma',
    'lang': 'en'
}
nlp = stanza.Pipeline(**config)

def preprocess_data(data):
    dictionary_list = []
    for idx, row in data.iterrows():
        tokens, pos, ner, head, deprel = [], [], [], [], []
        text_sample = row['samples']
        e11, e12, e21, e22 = int(row['e11']), int(row['e12']), int(row['e21']), int(row['e22'])

        doc = nlp(text_sample)
        for sentence in doc.sentences:
            tokens.extend([word.text for word in sentence.words])
            pos.extend([word.upos for word in sentence.words])
            head.extend([word.head for word in sentence.words])
            deprel.extend([word.deprel for word in sentence.words])
            ner.extend([token.ner if token.ner == 'O' else token.ner.split('-')[-1] for token in sentence.tokens])

        dictionary_list.append({
            'id': str(row['sentence_id']),
            'relation': str(row['relations']),
            'token': tokens,
            'subj_start': e11,
            'subj_end': e12,
            'obj_start': e21,
            'obj_end': e22,
            'subj_type': ner[e11],
            'obj_type': ner[e21],
            'stanford_pos': pos,
            'stanford_ner': ner,
            'stanford_head': head,
            'stanford_deprel': deprel
        })
    return dictionary_list

def tag2id(df, type_id):
    base_tags = ['<PAD>', '<UNK>']
    unique_tags = df[type_id].unique() if type_id in ['subj_type', 'obj_type'] else list(
        set(itertools.chain.from_iterable(df[type_id]))
    )
    base_tags.extend(unique_tags)
    return {tag: idx for idx, tag in enumerate(base_tags)}

def create_constants(df):
    return {
        'SUBJ_NER_TO_ID': tag2id(df, 'subj_type'),
        'OBJ_NER_TO_ID': tag2id(df, 'obj_type'),
        'NER_TO_ID': tag2id(df, 'stanford_ner'),
        'POS_TO_ID': tag2id(df, 'stanford_pos'),
        'DEPREL_TO_ID': tag2id(df, 'stanford_deprel')
    }

def convert_file(path_to_csv, dataset):
    outpath = Path('')
    outpath.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(f'{path_to_csv}/{dataset}.csv')

    # Partition the data and preprocess each split
    partitions = {'train': 'train', 'val': 'dev', 'test': 'test'}
    for partition, filename in partitions.items():
        split_data = data[data['partition'] == partition]
        dictionary_list = preprocess_data(split_data)
        with open(outpath / f"{filename}.json", 'w') as f:
            json.dump(dictionary_list, f, cls=NumpyEncoder)
        print(f'{partition} data created')

if __name__ == "__main__":
    path_to_csv = ""
    dataset = "NYT10"
    convert_file(path_to_csv, dataset)
