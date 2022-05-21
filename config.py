import torch
import torch.nn as nn
from dataloader import Vocab

# set maxlen of a sequence,batch_size,teacher_for_ratio
max_len = 10 #set maxlen of sequences
batch_size = 64 #set batch size
teacher_for_ratio = 0.5 #set teacher_for_ratio
drop = nn.Dropout(0.5)
comma = Vocab.comma # set id of ,
dot = Vocab.dot # set id of 。
Pad = Vocab.Pad # set padding id
Sos = Vocab.Sos # set start of a sentence
#config = Config(s_maxlen,s_batch_size,s_teacher_for_ratio)

