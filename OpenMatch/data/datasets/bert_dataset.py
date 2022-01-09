from typing import List, Tuple, Dict, Any

import json
import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

class BertDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        tokenizer: AutoTokenizer,
        mode: str,
        query_max_len: int = 32,
        doc_max_len: int = 256,
        doc_size: int = 5,
        max_input: int = 1280000,
        task: str = 'ranking'
    ) -> None:
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._mode = mode
        self._query_max_len = query_max_len
        self._doc_max_len = doc_max_len
        self._seq_max_len = query_max_len + doc_max_len + 3
        self._doc_size = doc_size
        self._max_input = max_input
        self._task = task
        if self._seq_max_len > 512:
            raise ValueError('query_max_len + doc_max_len + 3 > 512.')

        if self._mode ==  'inf':
            self._id = False
            self._examples = []
            query = self._dataset['query']
            document = self._dataset['document']

            for doc_dic in document:
                doc = doc_dic['doc']
                label = doc_dic['label']
                line = {'query': query, 'doc': doc, 'label': int(label)}
                self._examples.append(line)
            
        else:
            if isinstance(self._dataset, str):
                self._id = False
                with open(self._dataset, 'r') as f:
                    self._examples = []
                    for i, line in enumerate(f):
                        if i >= self._max_input:
                            break
                        if self._mode != 'train' or self._dataset.split('.')[-1] == 'json' or self._dataset.split('.')[-1] == 'jsonl':
                            line = json.loads(line)
                        else:
                            ####
                            if self._task == 'global' or self._task == 'global_no_att':
                                query, doc, label = line.strip('\n').split('\t')
                                line = {'query': query, 'doc': doc, 'label': int(label)}
                            ####
                            ####
                            elif self._task == 'global_cat':
                                lines = line.strip('\n').split('\t')
                                query = lines[0]
                                for j in lines[1:]:
                                    print('1')
                            ####
                            else:
                                raise ValueError('Task must be `ranking` or `classification`.')
                        self._examples.append(line)
            elif isinstance(self._dataset, dict):
                self._id = True
                self._queries = {}
                with open(self._dataset['queries'], 'r') as f:
                    for line in f:
                        if self._dataset['queries'].split('.')[-1] == 'json' or self._dataset['queries'].split('.')[-1] == 'jsonl':
                            line = json.loads(line)
                        else:
                            query_id, query = line.strip('\n').split('\t')
                            line = {'query_id': query_id, 'query': query}
                        self._queries[line['query_id']] = line['query']
                self._docs = {}
                with open(self._dataset['docs'], 'r') as f:
                    for line in f:
                        if self._dataset['docs'].split('.')[-1] == 'json' or self._dataset['docs'].split('.')[-1] == 'jsonl':
                            line = json.loads(line)
                        else:
                            doc_id, doc = line.strip('\n').split('\t')
                            line = {'doc_id': doc_id, 'doc': doc}
                        self._docs[line['doc_id']] = line['doc']
                if self._mode == 'dev':
                    qrels = {}
                    with open(self._dataset['qrels'], 'r') as f:
                        for line in f:
                            line = line.strip().split()
                            if line[0] not in qrels:
                                qrels[line[0]] = {}
                            qrels[line[0]][line[2]] = int(line[3])
                with open(self._dataset['trec'], 'r') as f:
                    self._examples = []
                    for i, line in enumerate(f):
                        if i >= self._max_input:
                            break
                        line = line.strip().split()
                        if self._mode == 'dev':
                            if line[0] not in qrels or line[2] not in qrels[line[0]]:
                                label = 0
                            else:
                                label = qrels[line[0]][line[2]]
                        if self._mode == 'train':
                            ###
                            if self._task == 'global' or self._task == 'global_no_att':
                                self._examples.append({'query_id': line[0], 'doc_id': line[1], 'label': int(line[2])})
                            ###
                            ###
                            elif self._task == 'global_cat':
                                print('1')
                            ###
                            else:
                                raise ValueError('Task must be `ranking` or `classification`.')
                        elif self._mode == 'dev':
                            self._examples.append({'label': label, 'query_id': line[0], 'doc_id': line[2], 'retrieval_score': float(line[4])})
                        elif self._mode == 'test':
                            self._examples.append({'query_id': line[0], 'doc_id': line[2], 'retrieval_score': float(line[4])})
                        else:
                            raise ValueError('Mode must be `train`, `dev` or `test`.')
            else:
                raise ValueError('Dataset must be `str` or `dict`.')
        self._count = len(self._examples)

    def collate(self, batch: Dict[str, Any]):
        if self._mode == 'train':
            ###
            if self._task == 'global' or self._task == 'global_no_att':
                input_ids = torch.tensor([item['input_ids'] for item in batch])
                segment_ids = torch.tensor([item['segment_ids'] for item in batch])
                input_mask = torch.tensor([item['input_mask'] for item in batch])
                label = torch.tensor([item['label'] for item in batch])
                return {'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': input_mask, 'label': label}
            ###
            ###
            elif self._task == 'global_cat':
                # input_ids = torch.tensor([item['input_ids'] for item in batch])
                # segment_ids = torch.tensor([item['segment_ids'] for item in batch])
                # input_mask = torch.tensor([item['input_mask'] for item in batch])
                # label = torch.tensor([item['label'] for item in batch])
                input_ids = torch.tensor([item['input_ids'] for item in batch])
                segment_ids = torch.tensor([item['segment_ids'] for item in batch])
                input_mask = torch.tensor([item['input_mask'] for item in batch])
                label = torch.tensor([item['label'] for item in batch])
                dataset =  [item['dataset'] for item in batch]
                return {'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': input_mask, 'label': label, 'dataset': dataset}
            ###
            else:
                raise ValueError('Task must be `ranking` or `classification`.')
        elif self._mode == 'dev':
            query_id = [item['query_id'] for item in batch]
            doc_id = [item['doc_id'] for item in batch]
            label = [item['label'] for item in batch]
            retrieval_score = torch.tensor([item['retrieval_score'] for item in batch])
            input_ids = torch.tensor([item['input_ids'] for item in batch])
            segment_ids = torch.tensor([item['segment_ids'] for item in batch])
            input_mask = torch.tensor([item['input_mask'] for item in batch])
            return {'query_id': query_id, 'doc_id': doc_id, 'label': label, 'retrieval_score': retrieval_score,
                    'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': input_mask}
        elif self._mode == 'test':
            query_id = [item['query_id'] for item in batch]
            doc_id = [item['doc_id'] for item in batch]
            retrieval_score = torch.tensor([item['retrieval_score'] for item in batch])
            input_ids = torch.tensor([item['input_ids'] for item in batch])
            segment_ids = torch.tensor([item['segment_ids'] for item in batch])
            input_mask = torch.tensor([item['input_mask'] for item in batch])
            return {'query_id': query_id, 'doc_id': doc_id, 'retrieval_score': retrieval_score,
                    'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': input_mask}
        elif self._mode == 'inf':
            input_ids = torch.tensor([item['input_ids'] for item in batch])
            segment_ids = torch.tensor([item['segment_ids'] for item in batch])
            input_mask = torch.tensor([item['input_mask'] for item in batch])
            label = torch.tensor([item['label'] for item in batch])
            return {'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': input_mask, 'label': label} 
        else:
            raise ValueError('Mode must be `train`, `dev` or `test`.')

    def pack_bert_features(self, query_tokens: List[str], doc_tokens: List[str]):
        input_tokens = [self._tokenizer.cls_token] + query_tokens + [self._tokenizer.sep_token] + doc_tokens + [self._tokenizer.sep_token]
        input_ids = self._tokenizer.convert_tokens_to_ids(input_tokens)
        segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(doc_tokens) + 1)
        input_mask = [1] * len(input_tokens)

        padding_len = self._seq_max_len - len(input_ids)
        input_ids = input_ids + [self._tokenizer.pad_token_id] * padding_len
        input_mask = input_mask + [0] * padding_len
        segment_ids = segment_ids + [0] * padding_len

        assert len(input_ids) == self._seq_max_len
        assert len(input_mask) == self._seq_max_len
        assert len(segment_ids) == self._seq_max_len

        return input_ids, input_mask, segment_ids

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]
        if self._id:
            example['query'] = self._queries[example['query_id']]
            if self._mode == 'train' and self._task == 'ranking':
                example['doc_pos'] = self._docs[example['doc_pos_id']]
                example['doc_neg'] = self._docs[example['doc_neg_id']]
            else:
                example['doc'] = self._docs[example['doc_id']]
        if self._mode == 'train':
            ###
            if self._task == 'global' or self._task == 'global_no_att':
                query_tokens = self._tokenizer.tokenize(example['query'])[:self._query_max_len]
                doc_tokens = self._tokenizer.tokenize(example['doc'])[:self._seq_max_len-len(query_tokens)-3]
                
                input_ids, input_mask, segment_ids = self.pack_bert_features(query_tokens, doc_tokens)
                return {'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': input_mask, 'label': example['label']}
            ###
            ###
            elif self._task == 'global_cat':
                query_tokens = self._tokenizer.tokenize(example['query'])[:self._query_max_len]

                input_ids_list = []
                input_mask_list = []
                segment_ids_list = []
                label_list = []

                for i, line in enumerate(example['document']):

                    if i < self._doc_size:
                        doc_tokens = self._tokenizer.tokenize(line['doc'])[:self._seq_max_len-len(query_tokens)-3]
                    
                        input_ids, input_mask, segment_ids = self.pack_bert_features(query_tokens, doc_tokens)

                        input_ids_list.append(input_ids)
                        input_mask_list.append(input_mask)
                        segment_ids_list.append(segment_ids)
                        label_list.append(line['label'])

                dic = example
                dic['document'] = dic['document'][self._doc_size:]

                # print(np.shape(input_ids_list))
                # print(np.shape(input_mask_list))
                # print(np.shape(segment_ids_list))
                # input_ids = torch.cat(input_ids_list, dim=1)
                # print(np.shape(label_list))
                # print(label_list[0] == 1)

                # input_mask = torch.cat(input_mask_list)
                # segment_ids = torch.cat(segment_ids_list)
                # label = torch.cat(label_list)

                return {'input_ids': input_ids_list, 'segment_ids': segment_ids_list, 'input_mask': input_mask_list, 'label': label_list, 'dataset' : dic}
            ###
            else:
                raise ValueError('Task must be `ranking` or `classification`.')
        elif self._mode == 'dev':
            query_tokens = self._tokenizer.tokenize(example['query'])[:self._query_max_len]
            doc_tokens = self._tokenizer.tokenize(example['doc'])[:self._seq_max_len-len(query_tokens)-3]

            input_ids, input_mask, segment_ids = self.pack_bert_features(query_tokens, doc_tokens)
            return {'query_id': example['query_id'], 'doc_id': example['doc_id'], 'label': example['label'], 'retrieval_score': example['retrieval_score'],
                    'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids}
        elif self._mode == 'test':
            query_tokens = self._tokenizer.tokenize(example['query'])[:self._query_max_len]
            doc_tokens = self._tokenizer.tokenize(example['doc'])[:self._seq_max_len-len(query_tokens)-3]

            input_ids, input_mask, segment_ids = self.pack_bert_features(query_tokens, doc_tokens)
            return {'query_id': example['query_id'], 'doc_id': example['doc_id'], 'retrieval_score': example['retrieval_score'],
                    'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids}
        elif self._mode == 'inf':
            query_tokens = self._tokenizer.tokenize(example['query'])[:self._query_max_len]
            doc_tokens = self._tokenizer.tokenize(example['doc'])[:self._seq_max_len-len(query_tokens)-3]
            
            input_ids, input_mask, segment_ids = self.pack_bert_features(query_tokens, doc_tokens)
            return {'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': input_mask, 'label': example['label']}
        else:
            raise ValueError('Mode must be `train`, `dev` or `test`.')

    def __len__(self) -> int:
        return self._count
