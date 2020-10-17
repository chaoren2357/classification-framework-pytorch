"""
Training script for Classification
Copyright (c) Zhiyuan GAO, 2020
"""

import argparse
import os
import time
import shutil

import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

import torchvision

import models
from utils import Bar, AverageMeter, accuracy, Logger, trans

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Classification in pytorch')

parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (''default: 4)')
parser.add_argument('-n', '--num_classes', default=10, type=int, help='number of classes (default: 10)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N', help='train batchsize (default: 128)')
parser.add_argument('--test-batch', default=100, type=int, metavar='N', help='test batchsize (default: 100)')

## Model options
parser.add_argument('-a', '--arch', metavar='ARCH', default='alexnet', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet20)')

## Optimization options
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float, metavar='LR',
                    help='initial learning rate (default: 0.1)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum (default: 0.9)')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')

## Device options
parser.add_argument('-gpu', '--use_gpu', default=False, type=bool)
parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

## Checkpoint options
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')

## Misc options
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

common_dataset = ['cifar10']
log = Logger()
best_acc = 0


def main():
    ############################### Preparing dataset ###############################
    global best_acc

    ## Transform for training image and test image
    transform_train, transform_test = trans(args.dataset)

    ## Define dataloader and num_classes
    if args.dataset in common_dataset:
        if args.dataset == 'cifar10':
            dataloader = torchvision.datasets.CIFAR10
            num_classes = 10
        elif args.dataset == 'cifar100':
            dataloader = torchvision.datasets.CIFAR100
            num_classes = 100
        else:  # Change
            assert "No way!"
            dataloader = None
            num_classes = -1
        trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

        testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
        testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    else:  # Change
        num_classes = -1
        trainloader = None
        testloader = None
        pass

    log.prepare_dataset(args.dataset, num_classes)
    ############################### Define model ###############################

    model = models.__dict__[args.arch](num_classes=num_classes)

    ## Choosing device
    if args.use_gpu and torch.cuda.is_available():
        use_cuda = True
        log.choose_device(use_cuda, torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
    else:
        use_cuda = False
        log.choose_device(use_cuda)

    ## Criterion
    criterion = nn.CrossEntropyLoss()

    ## Optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    log.define_model(arch=args.arch, params=model.parameters(), criterion=criterion, optim=optimizer, lr=args.lr,
                     momentum=args.momentum, weight_decay=args.weight_decay)

    ############################### Training or Evaluating ###############################

    ## Checkpoints
    title = args.dataset + '_' + args.arch

    if title not in os.listdir('checkpoints'):
        os.mkdir(os.path.join('checkpoints', title))

    if args.resume:

        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'

        checkpoint = torch.load(args.resume)
        best_acc = checkpoint["best_acc"]
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        ckpt_path = args.checkpoints
        pass

    else:
        start_epoch = args.start_epoch
        ckpt_path = os.path.join('checkpoints', title, 'test' + '_' + time.strftime("%Y%m%d_%H%M%S", time.localtime()))
        os.mkdir(ckpt_path)
        log.ready_training(ckpt_path,
                           ['Epoch', 'Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    ## Evaluation
    # if args.evaluate:
    #     logging.info('\n Evaluation only')
    #     test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
    #     logging.info(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
    #     return

    ## Training or evaluating

    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, use_cuda)

        # append logger file
        log.append([epoch, state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=ckpt_path)

    # logger.close()
    # logger.plot()
    # savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)


def train(trainloader, model, criterion, optimizer, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        # print()
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = "({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}".format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg
        )
        bar.next()
    bar.finish()
    return losses.avg, top1.avg


def test(testloader, model, criterion, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Testing ', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        with torch.no_grad():
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(testloader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == '__main__':
    main()
