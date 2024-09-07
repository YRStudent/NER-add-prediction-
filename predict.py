import torch
from models import Bert_BiLSTM_CRF
from utils import NerDataset, PadBatch, VOCAB, tokenizer, tag2idx, idx2tag
from transformers import BertTokenizer
#from openpyxl import load_workbook
import pandas as pd


class CRF_(object):
    def __init__(self,best_model,bert_model,device='cpu'):
        self.device = torch.device(device)
        self.model = Bert_BiLSTM_CRF(tag2idx)
        self.model.load_state_dict(torch.load(best_model))
        self.model.to(device)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
    
    def process(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids).to('cpu').unsqueeze(0)#这一步就不需要学pred里面的再放进model里面了
        pred_tags = []
        with torch.no_grad():
            preidct_val = self.model.predict_tags(input_ids,mask = None)
        for i,label in enumerate(preidct_val[0]):
            if i != 0 and i!= len(preidct_val)-1:
                pred_tags.append(idx2tag[label])
        pred_tags = ['[CLS]'] + pred_tags + ['[SEP]']
        return pred_tags, tokens

    #预测实体，封装到列表
    '''def entity_data(self, sentence, predict_lables):
        entities = []
        pre_label = predict_lables[0]
        word = ""
        for i in range(len(sentence)):
            current_labels = predict_lables[i]
            if current_labels.startswith('B'):
                if pre_label[2:] is not current_labels[2:] and word != "":
                    entities.append(word)
                word = ""
                word += sentence[i]
                pre_label = current_labels
            elif current_labels.startswith('I'):
                word += sentence[i]
                pre_label = current_labels
            elif current_labels.startswith('O'):
                if pre_label[2:] is not current_labels[2:] and word != "":
                    entities.append(word)
                pre_label = current_labels
                word = ""
        if word != "":
            entities.append(word)
        return entities'''

    def entity_data(self, tokens, pred_tags):
        entities = []
        entity = None
        for idx, st in enumerate(pred_tags):
            if entity is None:
                if st.startswith('B'):
                    entity = {}
                    entity['start'] = idx
                else:
                    continue
            else:
                if st == 'O':
                    entity['end'] = idx
                    name = ''.join(tokens[entity['start'] : entity['end']])
                    entities.append(name)
                    entity = None
                elif st.startswith('B'):
                    entity['end'] = idx
                    name = ''.join(tokens[entity['start'] : entity['end']])
                    entities.append(name)
                    entity = {}
                    entity['start'] = idx
                else:
                    continue
        return entities



if __name__ == '__main__':
    #从表格中读取文本
    '''data = pd.read_excel(r"E:\python projects\试验文本-医疗.xlsx")
    print(data)'''
    sentence = '阿达帕宁凝胶可以与夫西地酸乳膏一起用吗？'
    best_model = r"E:\python projects\Bert-BiLSTM-CRF\NER-MEDICAL\CCKS_2019_Task1\model_epoch_15.pth"
    bert_model = 'bert-base-chinese'
    crf = CRF_(best_model,bert_model,'cpu')
    pred_tags,tokens = crf.process(sentence)
    entities = crf.entity_data(tokens,pred_tags)
    #print(entities)
    type_list = []
    for i in pred_tags:
        if len(i) > 1:
            type_list.append(i[2:])
    print(type_list)
    #['延更丹', '大豆异黄胴']
#可能还需要加入F1、召回率、精准率等指标，这一块已经有了
#还需要测试实体-关系联合抽取的效果？不过我还是选择先实体抽取、再关系抽取吧

#print(pred_tags)
#['B-DRUG', 'I-DRUG', 'I-DRUG', 'O', 'O','O', 'B-DRUG', 'I-DRUG', 'I-DRUG', 'I-DRUG', 'I-DRUG', 'O', 'O', 'O', 'O', '[SEP]']

