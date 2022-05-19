import torch
import torch.nn as nn
import dataloader

# set maxlen of a sequence,batch_size,teacher_for_ratio
max_len = dataloader.maxlen #set maxlen of sequences
batch_size = 10 #set batch size
teacher_for_ratio = 0.5 #set teacher_for_ratio
drop = nn.Dropout(0.5)
comma = dataloader.comma # set id of ,
dot = dataloader.dot # set id of 。
Pad = dataloader.pad # set padding id
Sos = dataloader.sos # set start of a sentence
#config = Config(s_maxlen,s_batch_size,s_teacher_for_ratio)

