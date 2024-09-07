# -*- coding: utf-8 -*-
'''
@Author: Xavier WU
@Date: 2021-11-30
@LastEditTime: 2022-1-6
@Description: This file is for training, validating and testing. 
@All Right Reserve
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import os
import warnings
import argparse
import numpy as np
from sklearn import metrics
from models import Bert_BiLSTM_CRF
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import NerDataset, PadBatch, VOCAB, tokenizer, tag2idx, idx2tag
import json

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train(e, model, iterator, optimizer, scheduler, device):
    model.train()
    losses = 0.0
    step = 0
    for i, batch in enumerate(iterator):
        step += 1
        x, y, z = batch
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)

        loss = model(x, y, z)
        losses += loss.item()
        """ Gradient Accumulation """
        '''
          full_loss = loss / 2                            # normalize loss 
          full_loss.backward()                            # backward and accumulate gradient
          if step % 2 == 0:             
              optimizer.step()                            # update optimizer
              scheduler.step()                            # update scheduler
              optimizer.zero_grad()                       # clear gradient
        '''
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    print("Epoch: {}, Loss:{:.4f}".format(e, losses/step))

#原有validate函数
def validate(e, model, iterator, device):
    model.eval()
    Y, Y_hat = [], []
    losses = 0
    step = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            step += 1
            # x表示要预测的数据
            # y应该是答案？类似于想让模型预测的结果标签（单个）
            # z可能是一个额外的标签、掩码等或者是序列长度等权重...
            x, y, z = batch
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)

            # y_hat 表示预测的标签
            y_hat = model(x, y, z, is_test=True)

            loss = model(x, y, z)
            losses += loss.item()
            # Save prediction
            for j in y_hat:
              Y_hat.extend(j)
            # Save labels
            mask = (z==1)
            y_orig = torch.masked_select(y, mask)
            Y.append(y_orig.cpu())#Numpy函数还不支持在GPU张量上操作

    Y = torch.cat(Y, dim=0).numpy()
    Y_hat = np.array(Y_hat)
    acc = (Y_hat == Y).mean()*100 #这个计算的准确率 mean是均值计算

    y_true = [idx2tag[i] for i in Y]
    y_pred = [idx2tag[i] for i in Y_hat]

    #计算各值
    num_proposed = len(y_pred)
    num_correct = (y_true==y_pred).sum()
    #(np.logical_and(y_true==y_pred,y_true>1)).astype(np.int).sum()
    num_gold = len(y_true)

    try:
      precision = num_correct / num_proposed
    except ZeroDivisionError:
      precision = 1.0
    
    try:
      recall = num_correct / num_gold
    except ZeroDivisionError:
      recall = 1.0
    
    try:
      f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
      if precision*recall == 0:
        f1 = 1.0
      else:
        f1 = 0
    
    print("Epoch: {}, Val Loss:{:.4f}, Val Acc:{:.4f}, F1:{:.4f}, Recall:{:.4f}%".format(e, losses/step, acc, f1, recall))
    return model, losses/step, acc, f1, recall


def test(model, iterator, device):
  model.eval()
  Y, Y_hat = [], []
  with torch.no_grad():
      for i, batch in enumerate(iterator):
          x, y, z = batch
          x = x.to(device)
          z = z.to(device)
          y_hat = model(x, y, z, is_test=True)
          # Save prediction
          for j in y_hat:
            Y_hat.extend(j)
          # Save labels
          mask = (z==1).cpu()
          y_orig = torch.masked_select(y, mask)
          Y.append(y_orig)

  Y = torch.cat(Y, dim=0).numpy()
  y_true = [idx2tag[i] for i in Y]
  y_pred = [idx2tag[i] for i in Y_hat]

  return y_true, y_pred

if __name__=="__main__":

    labels = [
    'B-E-PLACE', 'B-E-EMOTION', 'B-E-PHMT-CLOTHES', 'B-E-PHMT-MAKEUP', 'B-E-PHMT-STAGE', 'B-E-CO-CHARACTER', 'B-E-CO-MUSIC', 'B-E-CO-SCRIPT', 'B-E-CO-TECH', 'B-E-IO-DOCUMENT', 'B-E-ACTOR-GROUP', 'B-E-ACTOR-PERSON', 'B-E-CO-TYPE-GENRE', 'B-E-CO-TYPE-SUBJECT', 'B-E-CO-TYPE-ROLE', 'B-E-CO-TYPE-IDENTITY', 'B-E-CO-TYPE-JOB', 'I-E-PLACE', 'I-E-EMOTION', 'I-E-PHMT-CLOTHES', 'I-E-PHMT-MAKEUP', 'I-E-PHMT-STAGE', 'I-E-CO-CHARACTER', 'I-E-CO-MUSIC', 'I-E-CO-SCRIPT', 'I-E-CO-TECH', 'I-E-IO-DOCUMENT', 'I-E-ACTOR-GROUP', 'I-E-ACTOR-PERSON', 'I-E-CO-TYPE-GENRE', 'I-E-CO-TYPE-SUBJECT', 'I-E-CO-TYPE-ROLE', 'I-E-CO-TYPE-IDENTITY', 'I-E-CO-TYPE-JOB']
    
    best_model = None
    _best_val_loss = 1e18#e代表10，表示10的18次方
    _best_val_acc = 1e-18#10的-18次方
    _best_val_f1 = 1e-9
    _best_val_recall = 1e-9


    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_epochs", type=int, default=40)
    parser.add_argument("--trainset", type=str, default="E:\\python projects\\YuOpera NER\\豫剧项目实操\\processed_data\\train_dataset.txt")
    parser.add_argument("--validset", type=str, default="E:\\python projects\\YuOpera NER\\豫剧项目实操\processed_data\\valid_dataset.txt")
    parser.add_argument("--testset", type=str, default= "E:\\python projects\\YuOpera NER\\豫剧项目实操\\processed_data\\test_dataset.txt")
    

    ner = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Bert_BiLSTM_CRF(tag2idx).cpu()

    print('Initial model Done.')
    train_dataset = NerDataset(ner.trainset)
    eval_dataset = NerDataset(ner.validset)
    test_dataset = NerDataset(ner.testset)
    print('Load Data Done.')

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=ner.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=PadBatch)

    eval_iter = data.DataLoader(dataset=eval_dataset,
                                 batch_size=(ner.batch_size)//2,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=PadBatch)

    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=(ner.batch_size)//2,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=PadBatch)

    #optimizer = optim.Adam(self.model.parameters(), lr=ner.lr, weight_decay=0.01)
    optimizer = AdamW(model.parameters(), lr=ner.lr, eps=1e-6)

    # Warmup
    len_dataset = len(train_dataset) 
    epoch = ner.n_epochs
    batch_size = ner.batch_size
    total_steps = (len_dataset // batch_size) * epoch if len_dataset % batch_size == 0 else (len_dataset // batch_size + 1) * epoch
    
    warm_up_ratio = 0.1 # Define 10% steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warm_up_ratio * total_steps, num_training_steps = total_steps)

    print('Start Train...,')
    checkpoint_history = []

    for epoch in range(1, ner.n_epochs+1):

        train(epoch, model, train_iter, optimizer, scheduler, device)
        candidate_model, loss, acc, f1, recall = validate(epoch, model, eval_iter, device)

        '''if os.path.exists(checkpoint_file):
          checkpoint_dict = load_checkpoint(checkpoint_file)
          best_acc = checkpoint_dict['best_acc']
          epoch_offset = checkpoint_dict['best_epoch']+1
          model.load_state_dict(torch.load())'''

        if loss < _best_val_loss and acc > _best_val_acc:
          best_model = candidate_model
          _best_val_loss = loss
          _best_val_acc = acc
          _best_val_f1 = f1
          _best_val_recall = recall
          # 这个是保存在每一个epoch中训练最优的模型参数
          torch.save(model.state_dict(),'model_epoch_{}.pth'.format(epoch))
          # 现在是要找所有最优模型文件中最优的文件
          checkpoint_history.append(
            {
            'epoch': epoch,
            'loss': _best_val_loss,
            'acc': _best_val_acc,
            'f1': _best_val_f1,
            'recall': _best_val_recall
          }
          )
          with open('checkpoint_dict.json','w') as f:
            json.dump(checkpoint_history, f, indent = 4)
        print("=============================================")
    
    y_test, y_pred = test(best_model, test_iter, device)
    print(metrics.classification_report(y_test, y_pred, labels=labels, digits=3))