import sys
import os

import warnings
from model import CSRNet
from utils import save_checkpoint
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from earlystoping import EarlyStopping
import numpy as np
import argparse
import json
import cv2
import dataset
import time
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch CSRNet utilize eddge with canny')

parser.add_argument('train_json', metavar='TRAIN',
                    help='path to train json')
parser.add_argument('test_json', metavar='TEST',
                    help='path to test json')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')

parser.add_argument('gpu',metavar='GPU', type=str,
                    help='GPU id to use.')

parser.add_argument('task',metavar='TASK', type=str,
                    help='task id to use.')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    
    global args,best_prec1
    
    best_prec1 = 1e6
    
    args = parser.parse_args()
    args.original_lr    = 1e-7
    args.lr             = 1e-10
    args.batch_size     = 1
    args.momentum       = 0.95
    args.decay          = 5*1e-4
    args.start_epoch    = 0
    args.epochs         = 200
    args.steps          = [-1,1,100,150]
    args.scales         = [1,1,1,1]
    args.workers        = 2
    args.seed           = time.time()
    args.print_freq     = 30
    with open(args.train_json, 'r') as outfile:        
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:       
        val_list = json.load(outfile)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)
    

    model = CSRNet()
    model = model.to(device)
    
    # loss function
    criterion = nn.MSELoss(reduction='sum').to(device)
    # optimizier
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                 momentum=args.momentum,
                                 weight_decay=args.decay)
    
    # if pretrained model exist
    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))
    
    # initialize early stopping
    early_stopping = EarlyStopping(patience=20, verbose=False)
    
    # loop for number of epoch
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        
        # invoke train function
        train(train_list, model, criterion, optimizer, epoch)
        # invoke validation
        prec1 = validate(val_list, model, criterion)
        
        # save best precision
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        # save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best,args.task)
        
        # call early stopping
        early_stopping(prec1)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
        
# define function for training
def train(train_list, model, criterion, optimizer, epoch):
    
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    # load data training via loader
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                       shuffle=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]), 
                       train=True, 
                       seen=model.seen,
                       batch_size=args.batch_size,
                       num_workers=args.workers),
        batch_size=args.batch_size)
    
    # set model to train mode
    model.train()
    end = time.time()
    
    # Add tqdm wrapper
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")

    for i, (img, target) in pbar:
        data_time.update(time.time() - end)
        
        img = img.to(device)
        img = Variable(img)
        output = model(img)
        
        target = target.type(torch.FloatTensor).unsqueeze(0).to(device)
        target = Variable(target)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        # compute loss
        loss = criterion(output, target)
        
        # gradient descent process
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        batch_time.update(time.time() - end)
        end = time.time()

        # Update tqdm bar with aditional information
        if i % args.print_freq == 0 or i == len(train_loader) - 1:
            pbar.set_postfix({
                'Loss': f'{losses.val:.4f}',
                'AvgLoss': f'{losses.avg:.4f}',
                'Time': f'{batch_time.val:.3f}s',
                'Data': f'{data_time.val:.3f}s'
            })

# define function for validation
def validate(val_list, model, criterion):
    print ('begin test')
    # load validation set
    test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
    batch_size=args.batch_size)    
    
    # set model to evaluation mode
    model.eval()
    
    mae = 0
    
    for i,(img, target) in enumerate(test_loader):
        img = img.to(device)
        img = Variable(img)
        output = model(img)
        
        # sum mae of each images
        mae += abs(output.data.sum()-target.sum().type(torch.FloatTensor).to(device))
    
    # computer mae 
    mae = mae/len(test_loader)    
    print(' * MAE {mae:.3f} '
              .format(mae=mae))

    return mae    

# define function to adjust learning rate    
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    args.lr = args.original_lr
    
    for i in range(len(args.steps)):
        
        scale = args.scales[i] if i < len(args.scales) else 1
        
        
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr

# define function to compute and stores the average and current value       
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
    
if __name__ == '__main__':
    main()        