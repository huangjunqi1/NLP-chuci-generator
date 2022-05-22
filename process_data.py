from nbformat import write
import numpy as np
#生成以空格分隔的带标点符号的句子，一行两句
def regular():
    with open('chuci.txt', 'r', encoding='utf-8-sig') as f:
        with open('after_process1.txt','w',encoding='utf-8-sig') as outf:
            i = 0
            for line in f:
                i += 1
                sents = line.split('，')
                print(sents,i)
                first_sent = list(sents[0]) 
                second_sent = list(sents[1])
                first_sent += '，'
                outf.write(''.join(first_sent))
                if(second_sent[-2] == '？' or second_sent[-2] == '！' or second_sent[-2] == '：' or second_sent[-2] =='?'):
                    second_sent[-2] = '。'
                elif(second_sent[-2]!='。'):
                    second_sent[-1] = '。'
                    second_sent.append('\n')
                outf.write(' ')
                outf.write(''.join(second_sent))

def create_dic():
    dic={}
    with open("resource/hjq1.txt",'r',encoding='utf-8-sig') as f:
        c=f.readlines()
        for line in c:
            # line= zhconv.convert(line[0], 'zh-hans')+line[1:]
            dic[line[0]]=line[2:-1]
    with open("resource/lyzed.txt",'r',encoding='utf-8-sig') as f:
        c=f.readlines()
        for line in c:
            # line=zhconv.convert(line[0], 'zh-hans')+line[1:]
            dic[line[0]]=line[2:-1]
    with open("resource/lmy1.txt",'r',encoding='utf-8-sig') as f:
        c=f.readlines()
        for line in c:
            # line=zhconv.convert(line[0], 'zh-hans')+line[1:]
            dic[line[0]]=line[2:-1]
    with open("resource/wqx.txt",'r',encoding='utf-8-sig') as f:
        c=f.readlines()
        for line in c:
            # line=zhconv.convert(line[0], 'zh-hans')+line[1:]
            dic[line[0]]=line[3:-1]
    print(dic)
    np.save('dict.npy', dic)
    return dic

def replace():
    dict = create_dic() #np.load('dict.npy').item()
    with open('after_process1.txt','r',encoding='utf-8-sig') as f:
        with open('after_replace.txt','w',encoding='utf-8-sig') as outf:
            for line in f:
                for word in line:
                    if(word == '\n'): continue
                    if(word in dict):
                        print(word,dict[word],end=' ')
                        word = dict[word]
                    outf.write(word)
                outf.write('\n')
#create_dic()
def ultimate_process(sent_num):
    with open('ultimate_data.txt','r',encoding='utf-8-sig') as f:
        with open('ultimate_chuci.txt','w',encoding='utf-8-sig') as outf:    
             i = 0
             for line in f:
                 sents = line.split(' ')
                 if(len(sents[0]) > 10 or len(sents[1]) > 11): continue
                 outf.write(sents[0])
                 second_sent = sents[1][:-1]
                 outf.write(second_sent) 
                 i += 2
                 if(i % sent_num == 0):outf.write('\n')

ultimate_process(8)               
# replace()