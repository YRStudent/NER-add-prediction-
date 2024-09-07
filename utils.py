# -*- coding: utf-8 -*-
'''
@Author: Xavier WU
@Date: 2021-11-30
@LastEditTime: 2022-1-6
@Description: This file is for implementing Dataset. 
@All Right Reserve
'''

#变成bert模型的输入模式
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

bert_model = 'bert base chinese'
tokenizer = BertTokenizer.from_pretrained(bert_model)
VOCAB = ('<PAD>', '[CLS]', '[SEP]', 'O', 'B-E-PLACE', 'B-E-EMOTION', 'B-E-PHMT-CLOTHES', 'B-E-PHMT-MAKEUP', 'B-E-PHMT-STAGE', 'B-E-CO-CHARACTER', 'B-E-CO-MUSIC', 'B-E-CO-SCRIPT', 'B-E-CO-TECH', 'B-E-IO-DOCUMENT', 'B-E-ACTOR-GROUP', 'B-E-ACTOR-PERSON', 'B-E-CO-TYPE-GENRE', 'B-E-CO-TYPE-SUBJECT', 'B-E-CO-TYPE-ROLE', 'B-E-CO-TYPE-IDENTITY', 'B-E-CO-TYPE-JOB', 'I-E-PLACE', 'I-E-EMOTION', 'I-E-PHMT-CLOTHES', 'I-E-PHMT-MAKEUP', 'I-E-PHMT-STAGE', 'I-E-CO-CHARACTER', 'I-E-CO-MUSIC', 'I-E-CO-SCRIPT', 'I-E-CO-TECH', 'I-E-IO-DOCUMENT', 'I-E-ACTOR-GROUP', 'I-E-ACTOR-PERSON', 'I-E-CO-TYPE-GENRE', 'I-E-CO-TYPE-SUBJECT', 'I-E-CO-TYPE-ROLE', 'I-E-CO-TYPE-IDENTITY', 'I-E-CO-TYPE-JOB')

#enumerate:返回对应的索引号，索引号在前[(0,x),(1,xx),(2,xxx)...]
tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}
MAX_LEN = 256 - 2

class NerDataset(Dataset):
    ''' Generate our dataset '''
    def __init__(self, f_path):
        self.sents = []
        self.tags_li = []

        with open(f_path, 'r', encoding = 'utf-8') as f:
            lines = [line.split('\n')[0] for line in f.readlines() if len(line.strip())!=0]

        words = [line.split('\t')[0] for line in lines]  
        tags =  [line.split('\t')[1] for line in lines]
        
        word, tag = [], []
        #zip上下对齐并封装在元组里
        #按照一句话结束的标志“。”去分割每个字和标签，并且加上特殊标记符
        for char, t in zip(words, tags):
            if char != '。':
                word.append(char)
                tag.append(t)
            else:
                if len(word) > MAX_LEN:
                  self.sents.append(['[CLS]'] + word[:MAX_LEN] + ['[SEP]'])
                  self.tags_li.append(['[CLS]'] + tag[:MAX_LEN] + ['[SEP]'])
                else:
                  self.sents.append(['[CLS]'] + word + ['[SEP]'])
                  self.tags_li.append(['[CLS]'] + tag + ['[SEP]'])
                word, tag = [], []

    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx]
        token_ids = tokenizer.convert_tokens_to_ids(words)
        laebl_ids = [tag2idx[tag] for tag in tags]
        seqlen = len(laebl_ids)
        return token_ids, laebl_ids, seqlen

    def __len__(self):
        return len(self.sents)
        
#对一个批次（batch）的数据进行填充（padding），以使得批次中所有序列的长度都达到该批次中最长序列的长度。
def PadBatch(batch):
    maxlen = max([i[2] for i in batch])
    token_tensors = torch.LongTensor([i[0] + [0] * (maxlen - len(i[0])) for i in batch])
    label_tensors = torch.LongTensor([i[1] + [0] * (maxlen - len(i[1])) for i in batch])
    #mask 用于指示哪些位置是有效的（非填充的）标记
    mask = (token_tensors > 0)
    return token_tensors, label_tensors, mask
