import os
import time
import logging
import argparse

import numpy as np

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from models import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='PreActResNet18')
    parser.add_argument('--first-dataset', default='cifar10')
    parser.add_argument('--second-dataset', default='mnist')
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument('--data-dir', default='./data', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--lr-one-drop', default=0.01, type=float)
    parser.add_argument('--lr-drop-epoch', default=100, type=int)
    parser.add_argument('--fname', default='eval_forgetting', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--chkpt-iters', default=1, type=int)
    return parser.parse_args()

args = get_args()

if args.first_dataset == 'cifar10':
    num_classes = 10
    in_size = 3*32*32
    data_mean = (0.4914, 0.4822, 0.4465)
    data_std = (0.2471, 0.2435, 0.2616)
elif args.first_dataset == 'svhn':
    num_classes = 10
    in_size = 3*32*32
    data_mean = (0.5, 0.5, 0.5)
    data_std = (0.5, 0.5, 0.5)
elif args.first_dataset == 'mnist':
    num_classes = 10
    in_size = 3*28*28
    data_mean = (0.1307, 0.1307, 0.1307)
    data_std = (0.3081, 0.3081, 0.3081)
mu = torch.tensor(data_mean).view(3,1,1).cuda()
std = torch.tensor(data_std).view(3,1,1).cuda()
def normalize(X):
    return (X - mu)/std

if args.second_dataset == 'cifar10':
    num_classes = 10
    in_size_second = 3*32*32
    data_mean = (0.4914, 0.4822, 0.4465)
    data_std = (0.2471, 0.2435, 0.2616)
elif args.second_dataset == 'svhn':
    num_classes = 10
    in_size_second = 3*32*32
    data_mean = (0.5, 0.5, 0.5)
    data_std = (0.5, 0.5, 0.5)
elif args.second_dataset == 'mnist':
    num_classes = 10
    in_size_second = 3*28*28
    data_mean = (0.1307, 0.1307, 0.1307)
    data_std = (0.3081, 0.3081, 0.3081)
mu_second = torch.tensor(data_mean).view(3,1,1).cuda()
std_second = torch.tensor(data_std).view(3,1,1).cuda()
def normalize_second(X):
    return (X - mu_second)/std_second

def main():
    proj_path = f"./logs"
    fname = f"{proj_path}/{args.fname}_{args.model}_{args.first_dataset}_{args.second_dataset}"
    if not os.path.exists(fname):
        os.makedirs(fname)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(fname, 'eval.log')),
            logging.StreamHandler()
        ])

    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.first_dataset == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])

        train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=test_transform)
    elif args.first_dataset == 'svhn':
        train_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
        train_dataset = datasets.SVHN(args.data_dir, split='train', download=True, transform=train_transform)
        test_dataset = datasets.SVHN(args.data_dir, split='test', download=True, transform=test_transform)
    elif args.first_dataset == 'mnist':
        train_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        train_dataset = datasets.MNIST(args.data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST(args.data_dir, train=False, download=True, transform=test_transform)
    else:
        raise ValueError("Unknown second dataset")
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.second_dataset == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])

        train_second_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=train_transform)
        test_second_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=test_transform)
    elif args.second_dataset == 'svhn':
        train_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
        train_second_dataset = datasets.SVHN(args.data_dir, split='train', download=True, transform=train_transform)
        test_second_dataset = datasets.SVHN(args.data_dir, split='test', download=True, transform=test_transform)
    elif args.second_dataset == 'mnist':
        train_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        train_second_dataset = datasets.MNIST(args.data_dir, train=True, download=True, transform=train_transform)
        test_second_dataset = datasets.MNIST(args.data_dir, train=False, download=True, transform=test_transform)
    else:
        raise ValueError("Unknown second dataset")
    train_second_loader = torch.utils.data.DataLoader(dataset=train_second_dataset, batch_size=args.batch_size, shuffle=True)
    test_second_loader = torch.utils.data.DataLoader(dataset=test_second_dataset, batch_size=args.batch_size, shuffle=False)

    start_epoch = 0

    if args.model == 'PreActResNet18':
        model = preactresnet.PreActResNet18(num_classes=num_classes)
    elif args.model == 'WideResNet':
        model = wideresnet.WideResNet(34, num_classes=num_classes, widen_factor=args.width_factor, dropRate=0.0)
    elif args.model == 'VGG11':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=False)
        model.classifier[-1] = nn.Linear(4096, num_classes)
    elif args.model == 'ResNet50':
        model = ResNet.ResNet50(num_classes=num_classes)
    elif args.model == 'DenseNet121':
        model = DenseNet.DenseNet121(num_classes=num_classes)
    else:
        raise ValueError("Unknown model")
    
    model = model.cuda()
    model.train()

    params = model.parameters()
    opt = torch.optim.SGD(params, lr=args.lr_max, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    best_test_acc = 0

    def lr_schedule(t):
        if t / args.epochs < 0.5:
            return args.lr_max
        elif t / args.epochs < 0.75:
            return args.lr_max / 10.
        else:
            return args.lr_max / 100.

    logger.info('Epoch \t Train Time \t Test Time \t LR \t \t Train Loss \t Train Acc \t Test Loss \t Test Acc  \t Train Set')
    for epoch in range(start_epoch, args.epochs):
        model.train()
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        
        train_set = ''
        if epoch / args.epochs < 0.5 or epoch / args.epochs >= 0.75:
            train_set = args.first_dataset
            for i, (X, y) in enumerate(train_loader):
                X, y = X.cuda(), y.cuda()
                
                lr = lr_schedule(epoch)
                opt.param_groups[0].update(lr=lr)

                output = model(normalize(X))
                loss = criterion(output, y)

                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss += loss.item() * y.size(0)
                train_acc += (output.max(1)[1] == y).sum().item()
                train_n += y.size(0)
        else:
            train_set = args.second_dataset
            for i, (X, y) in enumerate(train_second_loader):
                X, y = X.cuda(), y.cuda()
                
                lr = lr_schedule(epoch)
                opt.param_groups[0].update(lr=lr)

                output = model(normalize_second(X))
                loss = criterion(output, y)

                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss += loss.item() * y.size(0)
                train_acc += (output.max(1)[1] == y).sum().item()
                train_n += y.size(0)

        train_time = time.time()

        model.eval()
        test_loss = 0
        test_acc = 0
        test_n = 0
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()

            output = model(normalize(X))
            loss = criterion(output, y)

            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            test_n += y.size(0)

        test_time = time.time()

        logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %s',
            epoch, train_time - start_time, test_time - train_time, lr,
            train_loss/train_n, train_acc/train_n, test_loss/test_n, test_acc/test_n, train_set)

        # save checkpoint
        if (epoch+1) % args.chkpt_iters == 0 or epoch+1 == args.epochs:
            torch.save(model.state_dict(), os.path.join(fname, f'model_{epoch}.pth'))
            torch.save(opt.state_dict(), os.path.join(fname, f'opt_{epoch}.pth'))

        # save best
        if test_acc/test_n > best_test_acc:
            torch.save({
                    'state_dict':model.state_dict(),
                    'test_acc':test_acc/test_n,
                    'test_loss':test_loss/test_n,
                    'test_loss':test_loss/test_n,
                    'test_acc':test_acc/test_n,
                }, os.path.join(fname, f'model_best.pth'))
            best_test_acc = test_acc/test_n

if __name__ == "__main__":
    main()
