from __future__ import annotations
import zhconv
import json
import random
import torch
import config
from model import S2SModel
from dataloader import Vocab
import numpy as np
import copy
def get_dic(list):
    newdic={}
    # with open("data/dict.npy",'r', encoding='utf-8-sig') as f:
    dic=np.load('data/dict.npy', allow_pickle=True).item()
    for string in list:
        for word in string:
            if word in dic.keys() or zhconv.convert(word,'zh-hant') in dic.keys() :
                newdic[word] = dic[word]
        # print(newdic)
        # print(dic)
        return newdic


def get_real_chuci(list):
    dic={}
    # with open("mywork/dic.json",'r',encoding='utf-8-sig') as f:
    #     dic=json.load(f)
    dic=np.load('data/dict.npy', allow_pickle=True).item()
    ll=[]
    for line in list:
        string=''
        for word in line:
            convert=[]
            if word=='兮':
                string+=word
                continue
            for k,v in dic.items():
                if word==v:
                    convert.append(k)
            if len(convert)==0:
                string+=word
            else:
                c=random.random()
                if c>0.8:
                    string+=word
                else:
                    num=len(convert)
                    k=int(num*random.random())
                    string+=convert[k]
        ll.append(string)
    # print(ll)
    newdic=get_dic(ll)
    return ll,newdic

def generate(inputs):
    # print(Vocab.vocab['，'],Vocab.vocab['。'])
    # inputs: batch_size*num_sents*max_len    
    #outputs[:,sent_id,i,:] batch_size*num_sents*maxlen*voc_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'checkpoints\lvshi_best_model.pt'#f'checkpoints/{args.dataset}_{args.model}_best_model.pt'
    ckpt = torch.load(model_path)
    vocab = ckpt['vocab']
    inversed_vocab = ckpt['inversed_vocab']
    # 建立模型
    input_size = 300
    hidden_size = 512
    n_layers = 2
    model = S2SModel(
        voc_size=Vocab.vocab_size,
        input_size=input_size,
        hidden_size=hidden_size,
        n_layers=n_layers,
    )
    # 加载保存的参数到模型当中
    model.load_state_dict(ckpt['model'])
    model = model.to(device) 
    model.eval()
    outputs,hidden = model(inputs, teacher_force_ratio=0)
    sents=[]
    sent =''
    for i in range(inputs.size(1)):
        for j in range(0,config.max_len): #从第二个字到标点
            possiblity = outputs[0,i,j,:]            
            value,index = torch.topk(possiblity,5)
            if(index[0].item() == Vocab.vocab['，'] or index[0].item() ==  Vocab.vocab['。']): break
            word = inversed_vocab[index[0].item()]
            print(word)
            sent = sent + word
        sent += '，'if i%2 == 0 else '。'
        sents.append(copy.deepcopy(sent))
        sent = ''
        sents,annotations = get_real_chuci(sents)       
    return sents,annotations