import os
from argparse import ArgumentParser
import config
import torch
import torch.nn as nn
from dataloader import PoemDataset
from model import S2SModel
import time
from torch.utils.data import DataLoader


def train_one_epoch(model, optimizer, train_loader, args, epoch):
    loss_f = nn.CrossEntropyLoss(ignore_index=config.Pad)
    model.train()
    total_loss = 0.0
    start_time = time.time()
    log_step = 20
    n_batch = len(train_loader)

    for i,(input, target) in enumerate(train_loader):
        input, target = input.to(args.device), target.to(args.device)
        output, hidden = model(input, targets=target)
        # 计算loss
        loss = loss_f(output.view(-1, output.size(-1)), target.view(-1))
        total_loss += loss.item()
        # 计算梯度
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) #梯度裁剪
        optimizer.step()
        # 每隔一定循环数输出loss,监控训练过程
        if i % log_step == 0 and i > 0:
            avg_loss = total_loss / log_step
            elapse = time.time() - start_time
            print('| epoch {:3d} | batch {:3d}/{:3d} | {:5.2f} ms/batch | loss {:5.2f} |'.format(
                epoch, i, n_batch, elapse * 1000 / log_step, avg_loss
            ))
            start_time = time.time()
            total_loss = 0.0


def evaluate(model, test_loader, args):
    loss_f = nn.CrossEntropyLoss(ignore_index=config.Pad)
    model.eval() #进入测试模式，停止更新参数与dropout
    total_loss = 0.0
    total_batch = 0

    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.to(args.device), target.to(args.device)
            output, hidden = model(input)
            loss = loss_f(output.view(-1, output.size(-1)), target.view(-1))
            total_loss += loss.item()
            total_batch += 1

    return total_loss / total_batch


# Main
def main():
    parser = ArgumentParser()  #命令行参数
    parser.add_argument("--dataset", default="wuyanlvshi", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--grad_clip", default=0.1, type=float)

    parser.add_argument("--input_size", default=300, type=int)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--n_layers", default=2, type=int)

    os.makedirs('checkpoints', exist_ok=True)
    args = parser.parse_args()  
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = f'data/{args.dataset}.txt'
    dataset = PoemDataset(data_path)

    model = S2SModel(
        voc_size=dataset.vocab_size,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
    )
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(dataset.train_set, batch_size=args.batch_size, shuffle=True) #打包成batch
    test_loader = DataLoader(dataset.test_set, batch_size=args.batch_size, shuffle=False)

    best_loss = float('inf')
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        train_one_epoch(model, optimizer, train_loader, args, epoch)
        val_loss = evaluate(model, test_loader, args)

        print('-' * 65)
        print('| epoch {:3d} | time: {:5.2f}s | validation loss {:5.2f} | '.format(
            epoch, (time.time() - epoch_start_time), val_loss))
        print('-' * 65)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({'model': model.state_dict(),
                        'vocab': dataset.vocab,
                        'inversed_vocab': dataset.inversed_vocab},    
                       f'checkpoints/{args.dataset}_best_model.pt')

    torch.save({'model': model.state_dict(),
                'vocab': dataset.vocab,
                'inversed_vocab': dataset.inversed_vocab},
               f'checkpoints/{args.dataset}_final_model.pt')


if __name__ == '__main__':
    main()