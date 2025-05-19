import os
from model import ConvNeXt
from utils import save_checkpoint

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from earlystoping import EarlyStopping
import numpy as np
import argparse
import json
import dataset
import time
from tqdm import tqdm
import torch.nn.functional as F

from pytorch_msssim import ssim

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, pred, target):
        # SSIM returns similarity (closer to 1 = better), so 1 - ssim is loss
        # pred_map_resized = cv2.resize(pred, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_CUBIC)
        return 1 - ssim(pred, target, data_range=target.max() - target.min(), size_average=self.size_average)
        
parser = argparse.ArgumentParser(description='PyTorch ConvNeXt attention training')

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
    
    best_prec1              = 1e6
    args                    = parser.parse_args()
    args.original_lr        = 1e-7
    args.lr                 = 1e-10
    args.batch_size         = 1
    args.momentum           = 0.95
    args.decay              = 5*1e-4
    args.start_epoch        = 0
    args.epochs             = 200
    args.steps              = [-1,1,100,150]
    args.scales             = [1,1,1,1]
    args.workers            = 2
    args.seed               = time.time()
    args.print_freq         = 30
    with open(args.train_json, 'r') as outfile:        
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:       
        val_list = json.load(outfile)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)
    

    model = ConvNeXt()
    
    model = model.to(device)
    
    criterion = nn.MSELoss(reduction='sum').to(device)
    ssim_loss = SSIMLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                 momentum=args.momentum,
                                 weight_decay=args.decay)

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
    
    early_stopping = EarlyStopping(patience=20, verbose=False)
    
    for epoch in range(args.start_epoch, args.epochs):
        
        adjust_learning_rate(optimizer, epoch)
        
        train(train_list, model, criterion, optimizer, epoch, ssim_loss)
        prec1 = validate(val_list, model, criterion)
        
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best,args.task)
        
        early_stopping(prec1)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
        

def train(train_list, model, criterion, optimizer, epoch, ssim_loss):
    
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                       shuffle=True,
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                   ]), 
                       train=True, 
                       batch_size=args.batch_size,
                       num_workers=args.workers),
        batch_size=args.batch_size)
    
    model.train()
    end = time.time()
    
    # add tqdm progress bar
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")

    for i, (img, target) in pbar:
        data_time.update(time.time() - end)
        
        img = img.to(device)
        img = Variable(img)
        output = model(img)

        target = target.type(torch.FloatTensor).to(device)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        elif target.dim() == 2:
            B = target.shape[0]
            size = int(target.shape[1] ** 0.5)
            target = target.view(B, 1, size, size)
        # Resize target to match output size
        if target.shape[2:] != output.shape[2:]:
            target = F.interpolate(target, size=output.shape[2:], mode='bilinear', align_corners=False)

        # Compute hybrid SSIM loss
        loss = 0.8 * criterion(output, target) + 0.2 * ssim_loss(output, target)
        
        
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        batch_time.update(time.time() - end)
        end = time.time()

        # Update tqdm bar with additional information
        if i % args.print_freq == 0 or i == len(train_loader) - 1:
            pbar.set_postfix({
                'Loss': f'{losses.val:.4f}',
                'AvgLoss': f'{losses.avg:.4f}',
                'Time': f'{batch_time.val:.3f}s',
                'Data': f'{data_time.val:.3f}s'
            })
    
def validate(val_list, model, criterion):
    print ('begin test')
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
        
        mae += abs(output.data.sum()-target.sum().type(torch.FloatTensor).to(device))
        
    mae = mae/len(test_loader)    
    print(' * MAE {mae:.3f} '
              .format(mae=mae))

    return mae    
        
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