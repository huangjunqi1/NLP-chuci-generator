import torch
import torch.nn as nn
import random
import config

Pad = config.Pad
Eos = config.Eos

class Encoder(nn.Module):
    def __init__(self, voc_size, input_size, hidden_size,n_layers):
        super.__init__()
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
        embedding = self.embedding(input)
        embedding = self.drop(embedding)
        output , hidden = self.lstm(embedding,hidden)
        output = self.layer_norm(output)
        output = self.drop(output)
        return output,hidden

class S2SModel(nn.Module):
    def __init__(self,voc_size,input_size,hidden_size,n_layers,,sep_id):
        super.__init__()
        self.voc_size = voc_size
        self.max_len = config.max_len
        self.encoder = Encoder(self.voc_size,input_size,hidden_size,n_layers)
        self.decoder = AttentionDecoder(self.voc_size,input_size,hidden_size,hidden_size,n_layers)
        self.sep_id = sep_id
    def forward(self,inputs,hidden=None,targets=None,teacher_force_ratio=config.teacher_for_ratio):
        num_sents = inputs.size(1)
        outputs = torch.zeros(inputs.size(0),inputs.size(1),self.max_len,self.voc_size,device=inputs.device)
        enc_hidden = None
        enc_outputs = None

        for sent_id in range(num_sents):
            if sent_id > 0:
                enc_outpus,enc_hidden = self.encoder(enc_inputs,enc_hidden)
            input = inputs[:,sent_id,0]
            enc_inputs = torch.zeros(inputs.size(0),self.max_len + 1, dtype = torch.long, device = inputs.device)
            enc_inputs[:,0] = input
            for i in range(config.max_len):
                output,hidden = self.encoder(input.unsqueeze(1),hidden,enc_outputs)
                outputs[:,sent_id,i,:] = output(:,0,:)
                input = (targets[:,sent_id,i] if targets is not None and random.random() < teacher_force_ratio else output.argmax(2))
                enc_inputs[:,i+1] = input

            sep_id = self.sep_id[sent_id % 2]
            input = torch.tensor(sep_id , dtype = torch.long, device = inputs.device).expand_as(input)
            enc_inputs[:,self.max_len] = input
            output,hidden = self.decoder(input.unsqueeze(1),hidden,enc_outputs) 

class Attention(nn.Module):
    def __init__(self,enc_dim,dec_dim):
        super().__init__()
        self.atten = nn.Linear(enc_dim+dec_dim,dec_dim)
        self.v = nn.Linear(dec_dim,1,bias=False)
    def forward(self,dec_hidden,enc_outputs):
        dec_hiden = dec_hidden.unsqueeze(1).expand_as(enc_outputs)
        energy = torch.tanh(self.atten(torch.cat([dec_hidden,enc_outputs],dim=2)))
        score = self.v(energy).squeeze(2)
        return torch.softmax(score,1)

class AttentionDecoder(nn.Module):
    def __init__(self,voc_size,input_size,enc_output_size,hidden_size,n_layers):
        super().__init__()
        self.embedding = nn.Embedding(num_embedding=voc_size,embedding_dim=input_size)
        self.drop = config.drop
        self.enc_output_size = enc_output_size
        self.lstm = nn.LSTM(
            input_size = input_size + enc_output_size
            hidden_size = hidden_size
            num_layers = n_layers
            batch_first = True
        )
        self.attention = Attention(enc_output_size,hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size,voc_size)
    def forward(self,input,dec_hidden = None, enc_outputs = None):
        if (len(input.size())==1): input.unsqueeze_(1)
        embedding = self.embedding(input)
        embedding = self.drop(embedding)
        if enc_outputs is not None:
            a = self.attention(dec_hidden[0][-1],enc_outputs)
            context = a.unsqueeze(1) @ enc_outputs
        else:
            context = torch.zeros(embedding.size(0),1,self.enc_output_size,device=embedding.device)
        output,dec_hidden = self.lstm(torch.cat([embedding,context],dim=2),dec_hidden)
        output = self.layer_norm(output)
        output = self.drop(output)
        output = self.fc(output)
        return output,dec_hidden

