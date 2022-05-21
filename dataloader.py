import torch
from collections import Counter
from sklearn.model_selection import train_test_split
maxlen = 10

class vocab_load(object):
    def __init__(self,min_freq=2):
        counter = Counter()
        with open(f'data/jueju.txt','r',encoding='tuf-8-fig') as f1:
            for line in f1:
                counter.update(line)
        with open(f'data/lvshi.txt','r',encoding='utf-8-sig') as f2:
            for line in f2:
                counter.update(line)
        with open(f'data/chuci.txt','r',encoding='utf-8-sig') as f3:
            for line in f3:
                counter.update(line)
        vocab = [k for k,v in counter.items() if v>min_freq]
        self.comma = vocab.get("，")
        self.dot = vocab.get("。")
        self.vocab_size = len(vocab) + 3
        self.SOS = len(vocab) + 2
        self.Pad = len(vocab) + 1
        self.unkid = 0
        self.vocab = dict(zip(vocab, range(1, self.vocab_size-1)))
        self.inversed_vocab = dict(zip(range(1, self.vocab_size-1), vocab))
Vocab = vocab_load()

class PoemDataset(object):
    def __init__(self, data_path, test_size=0.1):
        all_sents = []

        with open(data_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                all_sents.append(line)
        
        self.vocab=Vocab
        self.entire_set = self.data_process(all_sents)
        self.train_set, self.test_set = train_test_split(self.entire_set, test_size=test_size, shuffle=True, random_state=0)

    def data_process(self, poems):
        processed_data = []
        for i, poem in enumerate(poems):
            poem = poem.strip()
            num_sent = 0
            for j,word in enumerate(poem):
                if (word == '，') or (word == '。'): num_sent += 1
            numeric = torch.tensor([[Pad]*config.max_len]*num_sent)
            now = 0
            sent_id = 0
            for word in enumerate(poem):
                if (word == '，') or (word == '。'): 
                    now = 0
                    sent_id += 1
                numeric[sent_id][now] = self.vocab.get(word,0)
                now+=1
            processed_data.append((numeric,numeric))
        return processed_data
