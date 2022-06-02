import torch
from argparse import ArgumentParser
from model import S2SModel
from dataloader import Vocab
import config

parser = ArgumentParser()
parser.add_argument("--dataset", default="lvshi", type=str)
#parser.add_argument("--model", default="simple", type=str)
parser.add_argument("--sentsnum",default=10,type=int)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = f'checkpoints/{args.dataset}_best_model.pt'
# 读取模型参数和词表
ckpt = torch.load(model_path,map_location=device)
vocab = Vocab.vocab
inversed_vocab = Vocab.inversed_vocab
# 建立模型
input_size = 300
hidden_size = 512
n_layers = 2
model = S2SModel(
    voc_size=Vocab.vocab_size,
    input_size=input_size,
    hidden_size=hidden_size,
    n_layers=n_layers,
)
# 加载保存的参数到模型当中
model.load_state_dict(ckpt['model'])
model = model.to(device)
# 设置生成的诗句数量
n_sents = args.sentsnum

while True:
    model.eval()
    first_sent = input("Input first sent")
    flag = False
    for word in first_sent:
        if word not in vocab:
            print(word, 'is not in vocab')
            flag = True
    if flag:
        continue
    un_kid = 0
    with torch.no_grad():
        inputv = torch.zeros(1,n_sents,config.max_len,dtype=torch.long,device = device)
        for i in range(n_sents):
            for j in range(config.max_len):
                inputv[0][i][j] = config.Pad
        ii = 0
        tokenid = []
        for word in first_sent:
            tokenid.append(vocab.get(word))
            inputv[0][0][ii] = tokenid[ii]
            ii = ii + 1
        inputv[0][0][ii] = vocab.get("，")
        inputv.to(device)
        outputs,hidden = model(inputv)
        print(first_sent,",")
        for i in range(1,n_sents):
            ans = ''
            for j in range(config.max_len):
                tmp1,tmp2 = torch.topk(outputs[0][i-1][j],2)
                tt = tmp2[0].item()
                if (tt == 0): tt = tmp2[1].item()
                word = inversed_vocab[tt]
                ans+=inversed_vocab[tt]
                if (word == '。') or (word == '，'): break
                
            print(ans)
