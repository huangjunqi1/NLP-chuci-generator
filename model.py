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
#key_padding_mask: True if padding,False if not padding

#need to modify 
#teacher_for_ratio
#lr
#dropout
#enc_hidden be the first sentence when the 5th sentence
#every one day
class Encoder(nn.Module):
    def __init__(self, voc_size, input_size, hidden_size,n_layers):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=voc_size, embedding_dim=input_size , padding_idx=Pad)
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
    def __init__(self,voc_size,input_size,hidden_size,n_layers):
        super().__init__()
        self.voc_size = voc_size
        self.max_len = config.max_len
        self.encoder = Encoder(self.voc_size,input_size,hidden_size,n_layers)
        self.decoder = AttentionDecoder(self.voc_size,input_size,hidden_size,hidden_size,n_layers)
    def forward(self,inputs,hidden=None,targets=None,teacher_force_ratio=config.teacher_for_ratio):
        num_sents = inputs.size(1)
        batch_size = inputs.size(0)
        outputs = torch.zeros(batch_size,num_sents-1,self.max_len,self.voc_size,device=inputs.device)
        enc_hidden = None
        enc_outputs = None
        Key_padding_mask = torch.zeros(batch_size,self.max_len,dtype=torch.bool,device = inputs.device)
        for sent_id in range(num_sents):
            #if (sent_id == 5):
                #enc_inputs = enc_input0
                # enc_hidden = enc_hidden0
            if (sent_id > 0):
                enc_outputs,enc_hidden = self.encoder(enc_inputs,enc_hidden)
            input = torch.LongTensor([Sos]*batch_size)
            input = input.to(inputs.device)
            enc_inputs = torch.zeros(inputs.size(0),self.max_len, dtype = torch.long, device = inputs.device)
            flag = []
            key_padding_mask = Key_padding_mask
            Key_padding_mask = torch.zeros(batch_size,self.max_len,dtype=torch.bool,device = inputs.device)
            for i in range(batch_size): flag.append(False)
            for i in range(self.max_len):
                #f i==0: continue
                output,hidden = self.decoder(key_padding_mask.detach(),input.unsqueeze(1),hidden,enc_outputs)
                if (sent_id > 0): outputs[:,sent_id-1,i,:] = output[:,0,:]
                if (sent_id == 0): input = inputs[:,0,i]
                else:
                    input = (targets[:,sent_id-1,i] if targets is not None and random.random() < teacher_force_ratio else output.argmax(2).squeeze(1))
                enc_inputs[:,i] = input
                for j in range(batch_size):
                    if (flag[j]):
                        Key_padding_mask[j][i] = True
                    if (input[j].item() == Eos1 or input[j].item() == Eos2): flag[j] = True
            #if (sent_id == 0): 
                #enc_hidden0 = enc_hidden
                #enc_input0 = enc_inputs
                
        return outputs,hidden

class Attention(nn.Module):
    def __init__(self,enc_dim,dec_dim):
        super().__init__()
        self.atten = nn.Linear(enc_dim+dec_dim,dec_dim)
        self.v = nn.Linear(dec_dim,1,bias=False)
    def forward(self,dec_hidden,enc_outputs,key_padding_mask):
        dec_hidden = dec_hidden.unsqueeze(1).expand_as(enc_outputs)
        # energy: batch_sz * maxlen * hidden_sz
        energy = torch.tanh(self.atten(torch.cat([dec_hidden,enc_outputs],dim=2)))
        #print(energy)
        scores = self.v(energy).squeeze(2)
        score = scores.masked_fill(key_padding_mask,-1e9)
        # score: batch_sz * maxlen
        #score = scores.masked_fill(key_padding_mask,-1e9)
        #print(score.size())
        
        return torch.softmax(score,dim=1)

class AttentionDecoder(nn.Module):
    def __init__(self,voc_size,input_size,enc_output_size,hidden_size,n_layers):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=voc_size,embedding_dim=input_size,padding_idx = Pad)
        self.drop = nn.Dropout(0.5)
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
        if len(input.size())==1: 
            input.unsqueeze_(1)
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

