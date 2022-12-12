import argparse
import shutil
import os
import time
import torch
import logging as logger
import torch.nn as nn
from torch import autocast
from torch.cuda.amp import GradScaler
from models.VGG_models import vgg11
from functions import seed_all, build_ncaltech, build_dvscifar

parser = argparse.ArgumentParser(description='PyTorch Neuromorphic Data Augmentation')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 10)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--dset', default='nc101', type=str, metavar='N', choices=['nc101', 'dc10'],
                    help='dataset')
parser.add_argument('--model', default='vgg11', type=str, metavar='N', choices=[ 'vgg11'],
                    help='neural network architecture')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')
parser.add_argument('--seed', default=1000, type=int,
                    help='seed for initializing training. ')
parser.add_argument('-T', '--time', default=10, type=int, metavar='N',
                    help='snn simulation time (default: 2)')
parser.add_argument('--amp', action='store_true',
                    help='if use amp training.')
parser.add_argument('--nda', action='store_true',
                    help='if use neuromorphic data augmentation.')
args = parser.parse_args()


def train(model, device, train_loader, criterion, optimizer, epoch, scaler, args):
    running_loss = 0
    model.train()
    M = len(train_loader)
    total = 0
    correct = 0
    s_time = time.time()
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)

        if args.amp:
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                mean_out = outputs.mean(1)
                loss = criterion(mean_out, labels)
                scaler.scale(loss.mean()).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            outputs = model(images)
            mean_out = outputs.mean(1)
            loss = criterion(mean_out, labels)
            loss.mean().backward()
            optimizer.step()

        running_loss += loss.item()
        total += float(labels.size(0))
        _, predicted = mean_out.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    e_time = time.time()
    return running_loss / M, 100 * correct / total, (e_time-s_time)/60


@torch.no_grad()
def test(model, test_loader, device):
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        mean_out = outputs.mean(1)
        _, predicted = mean_out.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())

    correct = torch.tensor([correct]).cuda()
    total = torch.tensor([total]).cuda()
    final_acc = 100 * correct / total
    return final_acc.item()


if __name__ == '__main__':

    seed_all(args.seed)

    if args.dset == 'nc101':
        train_dataset, val_dataset = build_ncaltech(transform=args.nda)
        num_cls = 101
        in_c = 2
    elif args.dset == 'dc10':
        train_dataset, val_dataset = build_dvscifar(transform=args.nda)
        num_cls = 10
        in_c = 2
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    if args.model == 'vgg11':
        model = vgg11(in_c=in_c, num_classes=num_cls)
    else:
        raise NotImplementedError

    model.T = args.time
    model.cuda()
    device = next(model.parameters()).device

    scaler = GradScaler() if args.amp else None

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/256 * args.batch_size, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)
    print('start training!')
    for epoch in range(args.epochs):

        loss, acc, t_diff = train(model, device, train_loader, criterion, optimizer, epoch, scaler, args)
        print('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f},\t time elapsed: {}'.format(epoch, args.epochs, loss, acc,
                                                                                    t_diff))
        scheduler.step()
        facc = test(model, test_loader, device)
        print('Epoch:[{}/{}]\t Test acc={:.3f}'.format(epoch, args.epochs, facc))
