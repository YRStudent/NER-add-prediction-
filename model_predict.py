# -*- coding: utf-8 -*-
'''
@Author: Xavier WU
@Date: 2021-11-30
@LastEditTime: 2022-1-6
@Description: This file is for building model. 
@All Right Reserve
'''

import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF

class Bert_BiLSTM_CRF(nn.Module):
    #放参数
    def __init__(self, tag_to_ix, embedding_dim=768, hidden_dim=256):
        super(Bert_BiLSTM_CRF, self).__init__()#这个必写，可以理解为固定写法
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim 

        self.bert = BertModel.from_pretrained('E:\\文献下载\\bert base chinese', return_dict=False)
        #hidden_size表示输出矩阵特征数，num_layers是堆叠几层的LSTM，hidden_dim除以2是因为它是双向的LSTM
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim//2,
                            num_layers=2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.1)#我看一般论文里面用的是0.5
        #为了防止数据过拟合，也是放在隐藏层也就是全连接层的节点中，上为BiLSTM，下为全连接层
        self.linear = nn.Linear(hidden_dim, self.tagset_size)#全连接层（hidden2tag layer）size:(n,hidden_dim)
        self.crf = CRF(self.tagset_size, batch_first=True)#crf层
            
    def _get_features(self, sentence):
        with torch.no_grad():#理解为阻止梯度更新，说明这个模型已经训练好了？或者说节省内存，停止反向计算梯度
          embeds, _  = self.bert(sentence,return_dict=False)
        enc, _ = self.lstm(embeds)
        enc = self.dropout(enc) #防止过拟合
        feats = self.linear(enc)
        return feats#返回emission matrix

    #输入输出放forward
    #这个函数是在计算损失函数loss和发射分数的对数似然
    def forward(self, sentence, tags, mask, is_test=False):
        emissions = self._get_features(sentence) #要输入进CRF的东西 emission matrix
        if not is_test: # Training，return loss 
            loss=-self.crf.forward(emissions, tags, mask, reduction='mean')#这里计算损失函数是需要加上负号的
            return loss
        else: # Testing，return decoding
            decode=self.crf.decode(emissions, mask)
            return decode

    #这个函数是进行预测的
    def predict_tags(self, sentence, mask):
        with torch.no_grad():
            emissions = self._get_features(sentence)
            predicted_tags = self.crf.decode(emissions, mask)
        return predicted_tags

