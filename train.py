# -*- coding: utf-8 -*-
'''
@Author: Xavier WU
@Date: 2021-11-30
@LastEditTime: 2022-1-6
@Description: This file is for training, validating and testing. 
@All Right Reserve
'''
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
#from tensorflow.python.keras.utils.version_utils import training
from torch.utils import data
import os
import warnings
import argparse
import numpy as np
import sklearn
from models import Bert_BiLSTM_CRF
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import NerDataset, PadBatch, VOCAB, tokenizer, tag2idx, idx2tag
import json
from sklearn.metrics import classification_report
from early_stopping import EarlyStopping
from sklearn.model_selection import KFold,StratifiedKFold
from dataset_split import data_k_fold

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train(e, model, iterator, optimizer, scheduler, device):
    #scheduler 用来调整学习率的东西
    #iterator 数据迭代器 通常是一个Dataloader对象，用于批量加载数据
    model.train()
    losses = 0.0#记录每个batch的损失
    step = 0#记录batch数量
    for i, batch in enumerate(iterator):
        #i指代当前batch索引
        #batch:从iterator中加载一个batch
        step += 1
        x, y, z = batch
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)

        #计算loss,is_test为默认Fasle
        loss = model(x, y, z)
        losses += loss.item() #将Pytorch的张量转换为python标量

        loss.backward()#反向传播 带着误差值去更新模型的超参数来减少未来预测的误差
        optimizer.step()#依据梯度更新模型参数
        scheduler.step()#调整学习率
        optimizer.zero_grad()#清空上轮梯度，防止梯度累积

    Loss = losses / step
    print("Epoch: {}, Loss:{:.4f}".format(e, Loss))#算平均损失

    return Loss


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
            y_hat = model(x, y, z, is_test=True)#这里的is_test表示是进行loss计算还是进行预测

            loss = model(x, y, z)
            losses += loss.item()
            # Save prediction
            for j in y_hat:
              Y_hat.extend(j)
            # Save labels
            mask = (z==1).to(device)
            y_orig = torch.masked_select(y, mask)
            Y.append(y_orig.cpu())   #Numpy函数还不支持在GPU张量上操作

    Y = torch.cat(Y, dim=0).numpy()
    Y_hat = np.array(Y_hat)
    acc = (Y_hat == Y).mean()*100 #这个计算的准确率 mean是均值计算就是用来算除数的


    print("Epoch: {}, Val Loss:{:.4f}, Val Acc:{:.4f}".format(e, losses/step, acc))

    return model, losses/step, acc #这个需要换成F1吗？


def test(model, iterator, device):
  model.eval()
  Y, Y_hat = [], []
  with torch.no_grad():
      for i, batch in enumerate(iterator):
          x, y, z = batch
          x = x.to(device)
          y = y.to(device)
          z = z.to(device)
          y_hat = model(x, y, z, is_test=True)
          # Save prediction
          for j in y_hat:
            Y_hat.extend(j)
          # Save labels
          mask = (z==1).to(device)
          y_orig = torch.masked_select(y, mask)
          Y.append(y_orig)

  Y = torch.cat(Y, dim=0).cpu().numpy()
  y_true = [idx2tag[i] for i in Y]
  y_pred = [idx2tag[i] for i in Y_hat]

  return y_true, y_pred

if __name__=="__main__":

    labels = ['B-E-PLACE', 'B-E-CO-CHARACTER', 'B-E-CO-MUSIC', 'B-E-CO-SCRIPT', 'B-E-CO-TECH', 'B-E-IO-DOCUMENT', 'B-E-ACTOR-GROUP', 'B-E-ACTOR-PERSON', 'B-E-CO-TYPE-SUBJECT', 'B-E-CO-TYPE-ROLE', 'I-E-PLACE', 'I-E-CO-CHARACTER', 'I-E-CO-MUSIC', 'I-E-CO-SCRIPT', 'I-E-CO-TECH', 'I-E-IO-DOCUMENT', 'I-E-ACTOR-GROUP', 'I-E-ACTOR-PERSON', 'I-E-CO-TYPE-SUBJECT', 'I-E-CO-TYPE-ROLE']

    best_model = None
    _best_val_loss = 1e18#e代表10，表示10的18次方
    _best_val_acc = 1e-18#10的-18次方


    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)#64——>8——>6
    parser.add_argument("--lr", type=float, default=0.0003)#原先为 0.001——>0.0001——>0.0002
    parser.add_argument("--n_epochs", type=int, default=100)

    parser.add_argument("--trainset", type=str, default="/root/autodl-tmp/YuOpera NER/豫剧项目实操/processed_data/train_dataset.txt")
    parser.add_argument("--validset", type=str, default="/root/autodl-tmp/YuOpera NER/豫剧项目实操/processed_data/valid_dataset.txt")
    parser.add_argument("--testset", type=str, default= "/root/autodl-tmp/YuOpera NER/豫剧项目实操/processed_data/test_dataset.txt")
    ner = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 初始化模型
    model = Bert_BiLSTM_CRF(tag2idx).to(device)

    ## k-fold cross validation
    '''parser.add_argument("--trainset", type=str,
                        default="/root/autodl-tmp/YuOpera NER/Yu_dataset_k-fold/train_dataset.txt")
    parser.add_argument("--testset", type=str,
                        default="/root/autodl-tmp/YuOpera NER/Yu_dataset_k-fold/test_dataset.txt")


    ner = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 初始化模型
    model = Bert_BiLSTM_CRF(tag2idx).to(device)
    #print('prepare k-fold data')

    #kf = KFold(n_splits=5, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=5)
    # 用于存储每折的验证结果
    fold_accuracies = []

    training_texts,training_tags = np.array(data_k_fold(ner.trainset))
    testing_texts,testing_tags = np.array(data_k_fold(ner.testset))


    for fold, (train_idx, val_idx) in enumerate(skf.split(training_texts,training_tags)):
        print(f"Fold {fold + 1}:")

        # 分割训练集和验证集
        train_texts, val_texts = training_texts[train_idx], training_texts[val_idx]
        train_tags, val_tags = training_tags[train_idx], training_tags[val_idx]

        # 初始化模型
        model = Bert_BiLSTM_CRF(tag2idx).to(device)

        # 把数据搞成训练集需要的格式
        # (1)第一步转换Dataset
        train_dataset = NerDataset(train_texts, train_tags)
        valid_dataset = NerDataset(val_texts, val_tags)
        test_dataset = NerDataset(testing_texts, testing_tags)

        # (2)Dataloader
        train_iter = data.DataLoader(dataset=train_dataset,
                                     batch_size=ner.batch_size,
                                     shuffle=True,  # 原先这里为 False
                                     num_workers=4,
                                     collate_fn=PadBatch,
                                     drop_last=True)

        eval_iter = data.DataLoader(dataset=valid_dataset,
                                    batch_size=(ner.batch_size) // 2,
                                    shuffle=False,
                                    num_workers=4,
                                    collate_fn=PadBatch)

        test_iter = data.DataLoader(dataset=test_dataset,
                                    batch_size=(ner.batch_size) // 2,
                                    shuffle=False,
                                    num_workers=4,
                                    collate_fn=PadBatch)
        # 准备优化器和warmup
        optimizer = AdamW(model.parameters(), lr=ner.lr, eps=1e-6)

        # Warmup
        len_dataset = len(train_dataset)
        epoch = ner.n_epochs
        batch_size = ner.batch_size
        total_steps = (len_dataset // batch_size) * epoch if len_dataset % batch_size == 0 else (len_dataset // batch_size + 1) * epoch

        warm_up_ratio = 0.1  # Define 10% steps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_ratio * total_steps,num_training_steps=total_steps)
        #设定在一个步数值内，以（所处的步数/总预热步数）作为倍数
        # 其实即为（当前训练轮数/总的所需轮数），使得学习率逐渐向一开始设定的lr靠近

        print('Start Train...,')

        # early_stopping
        save_path = '/root/autodl-tmp/YuOpera NER/checkpoint_skf{}.pt'.format(fold+1)
        es = EarlyStopping(patience=10, verbose=True, path= save_path)
        validate_loss = []
        train_loss = []
        val_acc = []
        train_acc = []

        for epoch in range(1, ner.n_epochs + 1):

            Loss = train(epoch, model, train_iter, optimizer, scheduler, device)
            _, _, tc = validate(epoch, model, train_iter, device) #评估train的准确度 tc:train accuracy
            candidate_model, loss, vc = validate(epoch, model, eval_iter, device)

            validate_loss.append(loss)
            train_loss.append(Loss)
            train_acc.append(tc)
            val_acc.append(vc)

            # 记录每一折上 训练过程中 验证集上accuracy得分最高
            if vc >= _best_val_acc:
                _best_val_acc = vc

            es(loss, model)
            if es.early_stop:
                print('Early Stopping')
                break

        # 在验证集上评估-我们选取得分最高的
        fold_accuracies.append(_best_val_acc)
        print(f"Validation Accuracy for Fold {fold + 1}: {_best_val_acc}")

    # 最后用来计算k-fold的平均值和标准差
    print('k-fold average accuracy: %.3f,k-fold std: %.3f' % (np.mean(fold_accuracies), np.std(fold_accuracies)))'''

    # 加载最后的模型进行后续测试或评估
    '''model.load_state_dict(torch.load('/root/autodl-tmp/YuOpera NER/checkpoint_skf5.pt'))

    y_test, y_pred = test(model, test_iter, device)
    print(classification_report(y_test, y_pred, labels=labels, digits=3))'''


    train_dataset = NerDataset(ner.trainset)
    eval_dataset = NerDataset(ner.validset)
    test_dataset = NerDataset(ner.testset)
    print('Load Data Done.')

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=ner.batch_size,
                                 shuffle=True, #原先这里为 False
                                 num_workers=4,
                                 collate_fn=PadBatch,
                                 drop_last=True)#多添了一个True 使得训练更加稳定
    #shuffle:打乱每个epoch中的数据，提高模型的泛化能力，False是确保结果的可重复性
    #num_workers:使用4个子线程并行加载数据，来加速数据加载过程
    #collate_fn:用于合并批量数据 默认为将样本堆叠成张量，自定义的为用来处理序列填充或复杂的数据结构
    #drop_last 就是将不足一个批次的数据给忽略，这样保证了参数更新及模型收敛的稳定性——>就是保证了训练更加稳定，而验证集和测试集他们不用反向传播，因此就不需要这样的大小统一

    eval_iter = data.DataLoader(dataset=eval_dataset,
                                 batch_size=(ner.batch_size)//2,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=PadBatch)
    #batch_size除以2是为了解决显存占用或性能优化的问题
    #因为在训练阶段，还需要进行反向传播，所以验证和测试集应当减小显存占用空间

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
    #early_stopping
    es = EarlyStopping(patience = 7, verbose = True, path = '/root/autodl-tmp/YuOpera NER/checkpoint_3e-4_size=8.pt')

    validate_loss = []
    train_loss = []
    val_acc = []
    train_acc = []




    for epoch in range(1, ner.n_epochs+1):

        Loss = train(epoch, model, train_iter, optimizer, scheduler, device)
        _, _, tc = validate(epoch, model, train_iter, device)
        candidate_model, loss, vc = validate(epoch, model, eval_iter, device)

        validate_loss.append(loss)
        train_loss.append(Loss)
        train_acc.append(tc)
        val_acc.append(vc)

        es(loss,model)
        if es.early_stop:
            print('Early Stopping')
            break

    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Train_Loss", color="#0000FF")
    plt.plot(range(1, len(validate_loss) + 1), validate_loss, label="Val_Loss", color="#CE6090")
    plt.plot(range(1, len(train_acc) + 1), train_acc, label="Train_Accuracy", color="#F9A31A")
    plt.plot(range(1, len(val_acc) + 1), val_acc, label="Val_Accuracy", color="#79C5CE")
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    #加载最后的模型进行后续测试或评估
    model.load_state_dict(torch.load('/root/autodl-tmp/YuOpera NER/checkpoint_3e-4_size=8.pt'))

    y_test, y_pred = test(model, test_iter, device)
    print(classification_report(y_test, y_pred, labels=labels, digits=3))