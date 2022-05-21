# url = 'https://hanyu.baidu.com/s?wd=%E4%BD%A0&ptype=zici'
# headers = {
#     'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.75 Safari/537.36'}
#
# import requests
#
# response = requests.get(url=url, headers=headers)
# #print(response.text)
# from urllib.parse import urlencode
# from bs4 import BeautifulSoup
# soup = BeautifulSoup(response.text, features='html.parser')
# soup.prettify()
# des=soup.find("dd")
# print(des)
# for child in des:
#     print(child.text)
#
import json as json
import os
import pandas
import zhconv
from tqdm import trange
import torch
from collections import Counter
with open('dic.json','r',encoding='utf-8-sig') as f:
    with open('dicc.txt','w',encoding='utf-8-sig') as outf:
        a = json.load(f)
        for key, value in a.items():
            outf.write(key)
            outf.write(': ')
            for ss in value:
                outf.write(ss)
            outf.write('\n')





# wd = '你'
# encode_res = urlencode({'k': wd}, encoding='utf-8')
# keyword = encode_res.split('=')[1]
# print(keyword)
# # 然后拼接成url
# url = 'https://www.baidu.com/s?wd=%s&pn=1' % keyword
# print(url)
# response = requests.get(url, headers={
#     'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.75 Safari/537.36'})
# res1 = response.text

