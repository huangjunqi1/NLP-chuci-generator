import json as json
import os
import pandas
import zhconv
from tqdm import trange
import torch
from collections import Counter
from sklearn.model_selection import train_test_split
S_CHUCI = 1
os.makedirs('data',exist_ok=True)
cnt={S_CHUCI:0}
fname ='chuci.json'
#poems = pandas.read_json('chuci.json')

with open('data/chuci.txt', 'w', encoding='utf-8-sig') as outf:
    with open(fname,'r',encoding='utf-8-sig') as f:
        a=json.load(f)
        for poet in a:
            con=poet['content']
            conv = [zhconv.convert(sent, 'zh-hans') for sent in con]
            for sent in con:
                outf.write(sent)
                outf.write('\n')
            cnt[S_CHUCI]+=1

data_path='data/chuci.txt'
class chuciDataset(object):
    def __init__(self,data_path,min_freq=1):
        counter=Counter()
        all_sents=[]
        self.sent_length=30
        self.unk_id=0

        with open(data_path,'r',encoding='utf-8-sig') as f:
            lines=f.readlines()
            for line in lines:
                counter.update(line)
                all_sents.append(line)
        with open('data/qiyanjueju.txt','r',encoding='utf-8-sig') as f:
            lines=f.readlines()
            for line in lines:
                counter.update(line)
                all_sents.append(line)
        with open('data/qiyanlvshi.txt','r',encoding='utf-8-sig') as f:
            lines=f.readlines()
            for line in lines:
                counter.update(line)
                all_sents.append(line)
        with open('data/wuyanjueju.txt','r',encoding='utf-8-sig') as f:
            lines=f.readlines()
            for line in lines:
                counter.update(line)
                all_sents.append(line)
        with open('data/wuyanlvshi.txt','r',encoding='utf-8-sig') as f:
            lines=f.readlines()
            for line in lines:
                counter.update(line)
                all_sents.append(line)
        vocab=[k for k,v in counter.items() if v<2]
        self.vocab_size=len(vocab)+1
        self.vocab=dict(zip(vocab,range(1,self.vocab_size)))
        self.inversed_vocab = dict(zip(range(1,self.vocab_size),vocab))

        self.entire_set=self.data_process(all_sents)

    # def data_process(selfself,poems):
    #     processed_data=[]
    #     for i,poem in enumerate(poems):
    #         poem=poem.strip()
    #         numeric=torch.tensor([self.vocab.get(word,self.unk_id) for word in poem],dtype=torch.long)
    #         # get() equals to [] mostly the only difference is get() continuing running when the key isnot exist
    #         numeric=numeric.view(-1,30)
    #         source=numeric[:,:-1]# although this guy seems useless
    #         target=numeric[:,1:]
    #         processed_data.append((source,target))


chuciDataset(data_path)
