import json as json
import os
import pandas
#import zhconv
from tqdm import trange
import torch
from urllib.parse import urlencode
import requests
#from bs4 import BeautifulSoup
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
            for sent in con:
                outf.write(sent)
                outf.write('\n')
            cnt[S_CHUCI]+=1

data_path='data/chuci.txt'



def zhushi(vocab):
    dic={}
    a=0
    for word in vocab:
        wd = word
        word = zhconv.convert(word, 'zh-hans')
        encode_res = urlencode({'k': wd}, encoding='utf-8')
        keyword = encode_res.split('=')[1]
        print(keyword)
        url='https://hanyu.baidu.com/s?wd=%s&ptype=zici' % word
        print(url)
        try:
            response = requests.get(url=url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.75 Safari/537.36'})
        except:
            a+=1
            continue
        soup=BeautifulSoup(response.text, features='html.parser')
        soup.prettify()
        des = soup.find("dd")
        dic[word]=[]
        if des:
            for child in des:
                if(child.text=='\n'):
                    continue
                dic[word].append(child.text)
    print(dic)
    print(a)
    with open("dic.json",'w') as f:
        f.write(json.dumps(dic))
    # with open ("dicc.txt",'w') as f:
    #


class chuciDataset(object):
    def __init__(self,data_path,min_freq=1):
        counter=Counter()
        chucicounter=Counter()
        chuci_sents=[]
        all_sents=[]
        self.sent_length=30
        self.unk_id=0

        with open(data_path,'r',encoding='utf-8-sig') as f:
            lines=f.readlines()
            for line in lines:
                counter.update(line)
                chucicounter.update(line)
                all_sents.append(line)
                chuci_sents.append(line)
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
        vocab=[k for k,v in counter.items() if v<4]
        chucivocab=[k for k,v in chucicounter.items() if v<4]
        jiao=[k for k in vocab if k in chucivocab]
        t=0
        # for k in jiao:
        #     t=t+1
        #     if t==10:
        #         print('\n')
        #         t=0
        #     print(k,end=' ')
        zhushi(jiao)

#chuciDataset(data_path)


