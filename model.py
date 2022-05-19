import torch
import torch.nn as nn
import random
import config
import dataloader

#Pad `padding` id
#Eos `end of sentence` id
Pad = config.Pad
Eos1 = config.comma
Eos2 = config.dot
Sos = config.Sos
#key_padding_mask: True if padding,Flase if not padding

class Encoder(nn.Module):
    def __init__(self, voc_size, input_size, hidden_size,n_layers):
        super.__init__()
        # input_size : embedding_dim
        self.embedding = self.embedding = nn.Embedding(num_embeddings=voc_size, embedding_dim=input_size , padding_idx=Pad)
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.drop = config.drop
        self.lstm = nn.LSTM(
                input_size = input_size,
                hidden_size = hidden_size,
                num_layers = n_layers,
                batch_first = True
            )
        self.layer_norm = nn.LayerNorm(hidden_size)
    def forward(self,input,hidden=None):
        # input: batch_size*max_len
        # embedding: batch_size*maxlen*voc_size
        embedding = self.embedding(input)
        embedding = self.drop(embedding)
        # output: batch_size*maxlen_hidden_size
        output , hidden = self.lstm(embedding,hidden)
        output = self.layer_norm(output)
        output = self.drop(output)
        return output,hidden

class S2SModel(nn.Module):
    def __init__(self,voc_size,input_size,hidden_size,n_layers,sep_id):
        super.__init__()
        self.voc_size = voc_size
        self.max_len = config.max_len
        self.encoder = Encoder(self.voc_size,input_size,hidden_size,n_layers)
        self.decoder = AttentionDecoder(self.voc_size,input_size,hidden_size,hidden_size,n_layers)
        # self.sep_id = sep_id
    def forward(self,inputs,hidden=None,targets=None,teacher_force_ratio=config.teacher_for_ratio):
        # inputs: batch_size*num_sents*max_len
        num_sents = inputs.size(1)
        batch_size = inputs.size(0)
        outputs = torch.zeros(inputs.size(0),inputs.size(1),self.max_len,self.voc_size,device=inputs.device)
        enc_hidden = None
        enc_outputs = None
        for sent_id in range(num_sents):
            key_padding_mask = torch.zeors(batch_size,self.max_len,dtype=torch.bool)
            if (sent_id == 0):
                enc_inputs = inputs
                continue
            enc_outputs,enc_hidden = self.encoder(enc_inputs,enc_hidden)
            input = torch.LongTensor([[Sos]]*config.batch_size)
            enc_inputs = torch.zeros(inputs.size(0),self.max_len + 1, dtype = torch.long, device = inputs.device)
            enc_inputs[:,0] = input
            flag = []
            for i in range(batch_size): flag.append(0)
            for i in range(self.max_len):
                output,hidden = self.decoder(input.unsqueeze(1),hidden,enc_outputs)
                outputs[:,sent_id,i,:] = output[:,0,:]
                input = (targets[:,sent_id,i] if targets is not None and random.random() < teacher_force_ratio else output.argmax(2))
                for j in range(batch_size):
                    if (input[j].item() == Eos1 or input[j].item() == Eos2): 
                        flag[j] = 1
                    else:
                        if (flag[j]):
                            input[j] = Pad
                            key_padding_mask[j][i] = True
                enc_inputs[:,i+1] = input
            # output,hidden = self.decoder(key_padding_mask,input.unsqueeze(1),hidden,enc_outputs) 
        return outputs,hidden

class Attention(nn.Module):
    def __init__(self,enc_dim,dec_dim):
        super().__init__()
        self.atten = nn.Linear(enc_dim+dec_dim,dec_dim)
        self.v = nn.Linear(dec_dim,1,bias=False)
    def forward(self,dec_hidden,enc_outputs,key_padding_mask):
        dec_hidden = dec_hidden.unsqueeze(1).expand_as(enc_outputs)
        energy = torch.tanh(self.atten(torch.cat([dec_hidden,enc_outputs],dim=2)))
        #print(energy)
        scores = self.v(energy).squeeze(2)
        score = scores.masked_fill(key_padding_mask,-1e9)
        #print(score.size())
        
        return torch.softmax(score,1)

class AttentionDecoder(nn.Module):
    def __init__(self,voc_size,input_size,enc_output_size,hidden_size,n_layers):
        super().__init__()
        self.embedding = nn.Embedding(num_embedding=voc_size,embedding_dim=input_size,padding_idx = Pad)
        self.drop = config.drop
        self.enc_output_size = enc_output_size
        self.lstm = nn.LSTM(
                input_size = input_size + enc_output_size,
                hidden_size = hidden_size,
                num_layers = n_layers,
                batch_first = True,
            )
        self.attention = Attention(enc_output_size,hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size,voc_size)
    def forward(self,key_padding_mask,input,dec_hidden = None, enc_outputs = None):
        if (len(input.size())==1): input.unsqueeze_(1)
        embedding = self.embedding(input)
        embedding = self.drop(embedding)
        if enc_outputs is not None:
            a = self.attention(dec_hidden[0][-1],enc_outputs,key_padding_mask)
            context = a.unsqueeze(1) @ enc_outputs
        else:
            context = torch.zeros(embedding.size(0),1,self.enc_output_size,device=embedding.device)
        output,dec_hidden = self.lstm(torch.cat([embedding,context],dim=2),dec_hidden)
        output = self.layer_norm(output)
        output = self.drop(output)
        output = self.fc(output)
        return output,dec_hidden

