import os
from argparse import ArgumentParser
import config
import torch
import torch.nn as nn
from dataloader import PoemDataset
from model import S2SModel
from model import oldmodel
import time
from torch.utils.data import DataLoader
from dataloader import Vocab

torch.autograd.set_detect_anomaly(True)
def train_one_epoch(model, optimizer, train_loader, args, epoch , old_new):

    loss_f = nn.CrossEntropyLoss(ignore_index=config.Pad)
    model.train()
    total_loss = 0.0
    start_time = time.time()
    log_step = 50
    n_batch = len(train_loader)
    #print(n_batch)
    for i,(input, target1) in enumerate(train_loader):
        #print(i,"hahahahaha")
        if (old_new == "old"): target = input
        else: target = target1
        input, target = input.to(args.device), target.to(args.device)
        output, hidden = model(input, targets=target)
        #print(i,"hahahahaha")
        loss = loss_f(output.view(-1, output.size(-1)), target.view(-1))
        total_loss =total_loss + loss.item()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) #梯度裁剪
        optimizer.step()
        if i > 0 and i % log_step == 0:
            avg_loss = total_loss / log_step
            elapse = time.time() - start_time
            print('| epoch {:3d} | batch {:3d}/{:3d} | {:5.2f} ms/batch | loss {:5.2f} |'.format(
                epoch, i, n_batch, elapse * 1000 / log_step, avg_loss
            ))
            start_time = time.time()
            total_loss = 0.0


def evaluate(model, test_loader, args):
    loss_f = nn.CrossEntropyLoss(ignore_index=config.Pad)
    model.eval()
    total_loss = 0.0
    total_batch = 0

    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.to(args.device), target.to(args.device)
            output, hidden = model(input)
            loss = loss_f(output.view(-1, output.size(-1)), target.view(-1))
            total_loss = total_loss + loss.item()
            total_batch = total_batch + 1

    return total_loss / total_batch


# Main
def main():
    parser = ArgumentParser()  #命令行参数
    parser.add_argument("--dataset", default="lvshi", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--grad_clip", default=0.1, type=float)
    parser.add_argument("--model",default=None,type = str)
    parser.add_argument("--load",default = None,type = str)
    parser.add_argument("--input_size", default=300, type=int)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--n_layers", default=2, type=int)

    os.makedirs('checkpoints', exist_ok=True)
    args = parser.parse_args()  
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = f'data/{args.dataset}.txt'
    dataset = PoemDataset(data_path)

    if args.model != None:
        if args.model == "old": 
            model = oldmodel(
                voc_size=Vocab.vocab_size,
                input_size=args.input_size,
                hidden_size=args.hidden_size,
                n_layers=args.n_layers,
            )
            flag = "old"
        else:
            model = S2SModel(
                voc_size=Vocab.vocab_size,
                input_size=args.input_size,
                hidden_size=args.hidden_size,
                n_layers=args.n_layers,
            )
            flag = "new"
    else:
        model = S2SModel(
                voc_size=Vocab.vocab_size,
                input_size=args.input_size,
                hidden_size=args.hidden_size,
                n_layers=args.n_layers,
            )
        flag = "new"
    if(args.load != None):
        model_path = f'checkpoints/{args.model}_{args.load}_final_model.pt'
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['model'])
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #print(len(dataset.train_set))
    train_loader = DataLoader(dataset.train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset.train_set, batch_size=args.batch_size, shuffle=False)
    best_loss = float('inf')
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        train_one_epoch(model, optimizer, train_loader, args, epoch , flag)
        val_loss = evaluate(model, test_loader, args)

        print('-' * 65)
        print('| epoch {:3d} | time: {:5.2f}s | validation loss {:5.2f} | '.format(
            epoch, (time.time() - epoch_start_time), val_loss))
        print('-' * 65)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({'model': model.state_dict(),
                        'vocab': Vocab.vocab,
                        'inversed_vocab': Vocab.inversed_vocab},    
                       f'checkpoints/{args.model}_{args.dataset}_best_model.pt')
            
    torch.save({'model': model.state_dict(),
                        'vocab': Vocab.vocab,
                        'inversed_vocab': Vocab.inversed_vocab},    
                       f'checkpoints/{args.model}_{args.dataset}_final_model.pt')

if __name__ == '__main__':
    main()