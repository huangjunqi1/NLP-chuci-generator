import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self, voc_size, input_size, hidden_size,n_layers):
        super.__init__()
        self.embedding = self.embedding = nn.Embedding(num_embeddings=voc_size, embedding_dim=input_size)
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(0.5)
        self.lstm = nn.LSTM(
                input_size = input_size,
                hidden_size = hidden_size,
                num_layers = n_layers,
                batch_first = True
            )
        self.layer_norm = nn.LayerNorm(hidden_size)
    def forward(self,input,hidden=None):
        embedding = self.embedding(input)
        embedding = self.drop(embedding)
        output , hidden = self.lstm(embedding,hidden)
        output = self.layer_norm(output)
        output = self.drop(output)
        return output,hidden

class S2SModel(nn.Module):
    def __init__(self,voc_size,input_size,hidden_size,n_layers,max_length,sep_id):
        super.__init__()
        self.voc_size = voc_size
        self.max_len = max_length
        self.encoder = Encoder(voc_size,input_size,hidden_size,n_layers)
        self.sep_id = sep_id
    def forward(self,inputs,hidden=None,tagets=None,teacher_force_ratio=0.5):
        num_sents = inputs.size(1)
        outputs = torch.zeros(inputs.size(0),inputs.size(1),self.max_len,self.voc_size,device=inputs.device)
        enc_hidden = None
        enc_outputs = None

        for sent_id in range(num_sents):
            if sent_id > 0
                enc_outpus,enc_hidden = self.encoder(enc_inputs,enc_hidden)