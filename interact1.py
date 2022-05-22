import torch
from argparse import ArgumentParser
from model import S2SModel
from dataloader import Vocab
import config

#parser = ArgumentParser()
#parser.add_argument("--dataset", default="wuyanlvshi", type=str)
#parser.add_argument("--model", default="simple", type=str)
#args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'checkpoints\lvshi_best_model.pt'
# 读取模型参数和词表
ckpt = torch.load(model_path,map_location=device)
vocab = Vocab.vocab
inversed_vocab = Vocab.inversed_vocab
# 建立模型
input_size = 300
hidden_size = 512
n_layers = 2
#model_fn = Seq2seqModel if args.model == 'seq2seq' else SimpleModel
model = S2SModel(
    voc_size=Vocab.vocab_size,
    input_size=input_size,
    hidden_size=hidden_size,
    n_layers=n_layers,
)
# 加载保存的参数到模型当中
model.load_state_dict(ckpt['model'])
model = model.to(device)
# 设置生成的诗句数量和长度
n_sents = 10

while True:
    model.eval()
    # 读取藏头诗的头
    first_sent = input("Input first sent")
    #heads = input('Input heads: ')
    #if len(heads) != n_sents:
    #    print('Invalid input length')
    #    continue
    flag = False
    for word in first_sent:
        if word not in vocab:
            print(word, 'is not in vocab')
            flag = True
    if flag:
        continue
    un_kid = 0
    with torch.no_grad():
        inputv = torch.zeros(1,n_sents,config.max_len,dtype=torch.long)
        for i in range(n_sents):
            for j in range(config.max_len):
                inputv[0][i][j] = config.Pad
        ii = 0
        tokenid = []
        for word in first_sent:
            tokenid.append(vocab.get(word))
            inputv[0][0][ii] = tokenid[ii]
            ii = ii + 1
        inputv.to(device)
        outputs,hidden = model(inputv)
        for i in range(n_sents):
            ans = 
            for j in range(config.max_len):
                tmp1,tmp2 = torch.topk(outputs[0][i][j],2)
                tt = tmp2[0].item()
                if (tt == 0): tt = tmp2[1].item()
                word = inversed_vocab[tt]
                ans+=inversed_vocab[tt]
                if (word == '。') or (word == '，'): break
                
            print(ans)
        #anss = outputs[0].argmax(2)

    

    # TODO: 请补全生成的代码
