#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models import resnet34
from models.mobilenet_v2.mobilenet_v2 import MobileNetV2
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from tiny_imagenet import TinyImageNet200


def save_state(model, best_acc, flag='T'):
    print('==> Saving model ...')
    state = {
            'best_acc': best_acc,
            'state_dict': model.state_dict(),
            }
    for key in list(state['state_dict'].keys()):
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
    torch.save(state, 'models/model_' + flag + '_best.pth.tar')

def kl_div(p_logit, q_logit, T):
    p = F.softmax(p_logit / T, dim=-1)
    kl = torch.sum(p * (F.log_softmax(p_logit / T, dim=-1)
                                  - F.log_softmax(q_logit / T, dim=-1)), 1)
    return torch.mean(kl)

def dist_label(y, q):
    q = F.softmax(q, dim=-1)
    dist = torch.sum(torch.abs(q - y), 1)

    return torch.mean(dist)


def dist_gap(p_logit, q_logit, T):
    p = F.softmax(p_logit / T, dim=-1)
    q = F.softmax(q_logit / T, dim=-1)
    dist = torch.sum(torch.abs(q - p), 1)

    return torch.mean(dist)


def train(epoch):
    model_T.train()
    model_S.train()

    for batch_idx, (data, target) in enumerate(trainloader):
        
        # forwarding
        target_onehot = Variable((torch.zeros(data.size()[0], 200).cuda()).scatter_(1, target.view(target.size()[0], 1).cuda(), 1))
        data, target = Variable(data.cuda()), Variable(target.cuda())

        optimizer_T.zero_grad()
        output_T = model_T(data)

        optimizer_S.zero_grad()
        output_S = model_S(data)

        ps_y = dist_label(target_onehot, output_S.detach())
        pt_y = dist_label(target_onehot, output_T.detach())
        ps_pt = dist_gap(output_T.detach(), output_S.detach(), 1)

        epsilon = torch.exp(-1 * (pt_y / (pt_y + ps_y)))
        delta = ps_y - epsilon * pt_y

        if ps_pt > delta and pt_y < ps_y:
            loss_S = criterion(output_S, target) + \
                     kl_div(output_T.detach(), output_S, 1)

            loss_S.backward()
            optimizer_S.step()

        else:
            loss_T = criterion(output_T, target) + \
                     kl_div(output_S.detach(), output_T, 1)
            loss_S = criterion(output_S, target) + \
                     kl_div(output_T.detach(), output_S, 1)

            loss_T.backward()
            loss_S.backward()

            optimizer_T.step()
            optimizer_S.step()

        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss_S.item(),
                optimizer_S.param_groups[0]['lr']))
    return

def test_T():
    global best_acc_T
    model_T.eval()
    test_loss = 0
    correct = 0

    for data, target in testloader:
        data, target = Variable(data.cuda()), Variable(target.cuda())
                                    
        output = model_T(data)
        test_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    acc = 100. * correct / len(testloader.dataset)

    if acc > best_acc_T:
        best_acc_T = acc
        save_state(model_T, best_acc_T, flag='T')
    
    test_loss /= len(testloader.dataset)
    print('\nTest set: T: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 128., correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc_T))

    return acc

def test_S():
    global best_acc_S
    model_S.eval()
    test_loss = 0
    correct = 0

    for data, target in testloader:
        data, target = Variable(data.cuda()), Variable(target.cuda())

        output = model_S(data)
        test_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    acc = 100. * correct / len(testloader.dataset)

    if acc > best_acc_S:
        best_acc_S = acc
        save_state(model_S, best_acc_S, flag='S')

    test_loss /= len(testloader.dataset)
    print('\nTest set: S: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 128., correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc_S))

    return acc

def adjust_learning_rate(optimizer, epoch):
    update_list = [100, 140, 180]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']*0.1
    return

if __name__=='__main__':
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true',
            help='set if only CPU is available')
    parser.add_argument('--data', action='store', default='dataset-path',
            help='dataset path')
    parser.add_argument('--arch', action='store', default='resnet',
            help='the architecture for the network: nin')
    parser.add_argument('--lr', action='store', default='0.01',
            help='the intial learning rate')
    parser.add_argument('--pretrained', action='store', default=None,
            help='the path to the pretrained model')
    parser.add_argument('--evaluate', action='store_true',
            help='evaluate the model')
    args = parser.parse_args()
    print('==> Options:',args)

    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # prepare the data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = TinyImageNet200(root=args.data, train=True,
                               transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = TinyImageNet200(root=args.data, train=False,
                              transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # define the model
    print('==> building model',args.arch,'...')
    if args.arch == 'resnet':
        model_T = resnet34.ResNet34()
        model_S = MobileNetV2(num_classes=200, width_mult=1.4)
    else:
        raise Exception(args.arch+' is currently not supported')

    # initialize the model
    if not args.pretrained:
        print('==> Initializing model parameters ...')
        best_acc_T = 0
        for m in model_T.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                c = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, 1.0 / c)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data = m.weight.data.zero_().add(1.0)

        best_acc_S = 0
        for m in model_S.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                c = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, 1.0 / c)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data = m.weight.data.zero_().add(1.0)
    else:
        print('==> Load pretrained model form', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained)
        best_acc_T = pretrained_model['best_acc']
        model_T.load_state_dict(pretrained_model['state_dict'])

    if not args.cpu:
        model_T.cuda()
        model_T = torch.nn.DataParallel(model_T, device_ids=range(torch.cuda.device_count()))

        model_S.cuda()
        model_S = torch.nn.DataParallel(model_S, device_ids=range(torch.cuda.device_count()))
    print(model_S)

    # define solver and criterion
    base_lr = float(args.lr)
    #Teacher
    param_dict_T = dict(model_T.named_parameters())
    params_T = []
    for key, value in param_dict_T.items():
        params_T += [{'params':[value], 'lr': base_lr,
            'weight_decay':5e-4}]
        optimizer_T = optim.SGD(params_T, lr=args.lr, momentum=0.9, weight_decay=5e-4)

    #Student
    param_dict_S = dict(model_S.named_parameters())
    params_S = []
    for key, value in param_dict_S.items():
        params_S += [{'params': [value], 'lr': base_lr,
                    'weight_decay':5e-4}]
        optimizer_S = optim.SGD(params_S, lr=args.lr, momentum=0.9, weight_decay=5e-4)

    criterion = nn.CrossEntropyLoss()

    # start training

    for epoch in range(1, 200):
        adjust_learning_rate(optimizer_T, epoch)
        adjust_learning_rate(optimizer_S, epoch)

        train(epoch)

        acc_T = test_T()
        acc_S = test_S()





