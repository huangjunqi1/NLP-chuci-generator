import json as json
import os
import re
import pandas
import zhconv
from tqdm import trange
import torch
from collections import Counter
from sklearn.model_selection import train_test_split
def filter():
    os.makedirs('mywork', exist_ok=True)
    with open("data/chuci.txt", 'r', encoding='utf-8-sig') as f:
        with open("mywork/cced.txt", 'w', encoding='utf-8-sig') as outf:
            con = f.readlines()
            for orline in con:
                c = re.split('，|。|？|！|：', orline)
                # print(len(c))
                for line in c:
                    if len(c) == 3:
                        outf.write(line)
                        outf.write('\n')
                    else:
                        print(line)

                outf.write('\n')

def create_dic():
    dic={}
    with open("resource/hjq1.txt",'r',encoding='utf-8-sig') as f:
        c=f.readlines()
        for line in c:
            line= zhconv.convert(line[0], 'zh-hans')+line[1:]
            dic[line[0]]=line[2:-1]
    with open("resource/lyzed.txt",'r',encoding='utf-8-sig') as f:
        c=f.readlines()
        for line in c:
            line=zhconv.convert(line[0], 'zh-hans')+line[1:]
            dic[line[0]]=line[2:-1]
    with open("resource/lmy1.txt",'r',encoding='utf-8-sig') as f:
        c=f.readlines()
        for line in c:
            line=zhconv.convert(line[0], 'zh-hans')+line[1:]
            dic[line[0]]=line[2:-1]
    with open("resource/wqx.txt",'r',encoding='utf-8-sig') as f:
        c=f.readlines()
        for line in c:
            line=zhconv.convert(line[0], 'zh-hans')+line[1:]
            dic[line[0]]=line[3:-1]
    print(dic)
    return dic
def plus_change():
    n=0
    dic=create_dic()
    with open("mywork/cced.txt", 'r', encoding='utf-8-sig') as f:
        with open("mywork/ccedplus.txt", 'w', encoding='utf-8-sig') as outf:
            lines=f.readlines()
            for line in lines:
                if line == '\n':
                    continue
                n+=1
                for word in line:
                    if word in dic.keys():
                        word=dic[word]
                        print(1)
                if n%2==1:
                    line=line[0:-1]+'，'
                    outf.write(line)
                    outf.write('\n')
                else:
                    line=line[0:-1]+'。'
                    outf.write(line)
                    outf.write('\n')
                    outf.write('\n')

def get_dic(string):
    with open("dic.json",'r', encoding='utf-8-sig') as f:
        newdic={}
        dic=json.load(f)
        for word in string:
            if word in dic.keys():
                newdic[word]=dic[word]
        return newdic

get_dic('阰齌wdsfa')
