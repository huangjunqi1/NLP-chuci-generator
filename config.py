import torch
import torch.nn as nn


# set maxlen of a sequence,batch_size,teacher_for_ratio
max_len = 8 #set maxlen of sequences
batch_size = 10 #set batch size
teacher_for_ratio = 0.5 #set teacher_for_ratio
drop = nn.Dropout(0.5)
Eos = 0 # set EOS id
Pad = 1 # set padding id
#config = Config(s_maxlen,s_batch_size,s_teacher_for_ratio)

