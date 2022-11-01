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

from models import wrnet, wrnet2, mobilenet
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

def save_state(model, best_acc, flag=''):
    print('==> Saving model ...')
    state = {
            'best_acc': best_acc,
            'state_dict': model.state_dict(),
            }
    for key in list(state['state_dict'].keys()):
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
    torch.save(state, 'models/' + flag + '_best.pth.tar')


def kl_div(p_logit, q_logit, T):
    p = F.softmax(p_logit / T, dim=-1)
    kl = torch.sum(p * (F.log_softmax(p_logit / T, dim=-1)
                                  - F.log_softmax(q_logit / T, dim=-1)), 1)
    return torch.mean(kl)



def dist_s_label(y, q):

    q = F.softmax(q, dim=-1)
    dist = torch.sum(torch.abs(q - y), 1)
	
    return torch.mean(dist)


def dist_s_t(p_logit, q_logit, T):
    p = F.softmax(p_logit / T, dim=-1)
    q = F.softmax(q_logit / T, dim=-1)
    dist = torch.sum(torch.abs(q - p), 1)

    return torch.mean(dist)


def compute_entropy(p_logit, T):
    p = F.softmax(p_logit / T, dim=-1)
    entropy = torch.sum(p * (-F.log_softmax(p_logit / T, dim=-1)), 1)
    
    return torch.mean(entropy)



def train(epoch):
    model.train()
    model2.train()
    model_ks2.train()

    w1 = 1
    w2 = 1
    w0 = 1

    for batch_idx, (data, target) in enumerate(trainloader):


        target_onehot = Variable((torch.zeros(data.size()[0], 100).cuda()).scatter_(1, target.view(target.size()[0], 1).cuda(), 1))

        
        # forwarding
        data, target = Variable(data.cuda()), Variable(target.cuda())

        optimizer.zero_grad()
        output = model(data)

        optimizer2.zero_grad()
        output2 = model2(data)

        optimizer_ks2.zero_grad()
        output_ks2 = model_ks2(data)


        #full2
        s_label2 = dist_s_label(target_onehot, output_ks2.detach())
        t_label2 = dist_s_label(target_onehot, output2.detach())
        s_t2 = dist_s_t(output2.detach(), output_ks2.detach(), 1)

        #norm
        ps_pt2 = dist_s_t(output2.detach(), output_ks2.detach(), 1)

        epsilon2 = torch.exp(-1 * (t_label2 / (t_label2 + s_label2)) )
        delta2 = s_label2 - epsilon2 * t_label2


        s_label = dist_s_label(target_onehot, output.detach())
        t_label = dist_s_label(target_onehot, output2.detach())
        s_t = dist_s_t(output2.detach(), output.detach(), 1)

        #norm
        ps_pt = dist_s_t(output2.detach(), output.detach(), 1)

        epsilon = torch.exp(-1 * (t_label2 / (t_label2 + s_label2)) )
        delta = s_label - epsilon * t_label


        if ps_pt2 > delta2 and t_label2 < s_label2:
            w2 = 0
        else:
            w2 = 1
        if ps_pt > delta and t_label < s_label:
            w1 = 0
        else:
            w1 = 1
        if w1 == 0 and w2 == 0:
            w0 = 0
        else:
            w0 = 1


        loss_full2 = w0 * criterion(output2, target) + \
                    w2 * kl_div(output_ks2.detach(), output2, 1) + w1 * kl_div(output.detach(), output2, 1)

 
        loss_full2.backward()

        optimizer2.step()



        # backwarding
        loss_full = criterion(output, target) + \
                    kl_div(output_ks2.detach(), output, 1) + kl_div(output2.detach(), output, 1)

 
        loss_full.backward()

        optimizer.step()


        loss_ks2 = criterion(output_ks2, target) + \
                   kl_div(output.detach(), output_ks2, 1) + kl_div(output2.detach(), output_ks2, 1)


        loss_ks2.backward()

        optimizer_ks2.step()


        if batch_idx % 100 == 0:
            print(epoch)
            #print('T criterion:' + str(criterion(output, target).item()))
            #print('S criterion:' + str(criterion(output_ks2, target).item()) + ' kl_div:' + str(kl_div(output.detach(), output_ks2, 1).item()))

            #print('T entropy:' + str(compute_entropy(output, 1).item()))
            #print('S entropy:' + str(compute_entropy(output_ks2, 1).item()))


            print('s_label:' + str(s_label.item()))
            print('t_label:' + str(t_label.item()))
            print('s_t:' + str(s_t.item()))
            print('delta:' + str(delta.item()))

            print('s_label2:' + str(s_label2.item()))
            print('t_label2:' + str(t_label2.item()))
            print('s_t2:' + str(s_t2.item()))
            print('delta2:' + str(delta2.item()))



            
    return

def test():
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in testloader:
        data, target = Variable(data.cuda()), Variable(target.cuda())
                                    
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    acc = 100. * correct / len(testloader.dataset)

    if acc > best_acc:
        best_acc = acc
        save_state(model, best_acc, flag='wrn16-2')
    
    test_loss /= len(testloader.dataset)
    print('\nTest set: Full: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 128., correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))

    return acc



def test2():
    global best_acc2
    model2.eval()
    test_loss = 0
    correct = 0

    for data, target in testloader:
        data, target = Variable(data.cuda()), Variable(target.cuda())
                                    
        output = model2(data)
        test_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    acc = 100. * correct / len(testloader.dataset)

    if acc > best_acc2:
        best_acc2 = acc
        save_state(model2, best_acc2, flag='wrn16-10')
    
    test_loss /= len(testloader.dataset)
    print('\nTest set: Full2: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 128., correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc2))

    return acc


def test_ks2():
    global best_acc_ks2
    model_ks2.eval()
    test_loss = 0
    correct = 0

    for data, target in testloader:
        data, target = Variable(data.cuda()), Variable(target.cuda())

        output = model_ks2(data)
        test_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    acc = 100. * correct / len(testloader.dataset)

    if acc > best_acc_ks2:
        best_acc_ks2 = acc
        save_state(model_ks2, best_acc_ks2, flag='mobilenet')

    test_loss /= len(testloader.dataset)
    print('\nTest set: ks2: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 128., correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc_ks2))

    return acc



def adjust_learning_rate(optimizer, epoch):
    update_list = [140, 200, 250]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']*0.1
    return

if __name__=='__main__':
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true',
            help='set if only CPU is available')
    parser.add_argument('--data', action='store', default='/home/qianbiao/data/CIFAR100',
            help='dataset path')
    parser.add_argument('--arch', action='store', default='resnet',
            help='the architecture for the network: nin')
    parser.add_argument('--lr', action='store', default='0.001',
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
    '''
    if not os.path.isfile(args.data+'/train_data'):
        # check the data path
        raise Exception\
                ('Please assign the correct data path with --data <DATA_PATH>')
    '''

    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    train_dataset = torchvision.datasets.CIFAR100(
        root=args.data,
        train=True,
        download=False,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

    test_dataset = torchvision.datasets.CIFAR100(
        root=args.data,
        train=False,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    # define classes
    #classes = ('plane', 'car', 'bird', 'cat',
    #        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



    # define the model
    print('==> building model',args.arch,'...')
    if args.arch == 'resnet':
        model = wrnet.wrnet16()
        model2 = wrnet2.wrnet16()
        model_ks2 = mobilenet.MobileNet()
    else:
        raise Exception(args.arch+' is currently not supported')

    # initialize the model
    if not args.pretrained:
        print('==> Initializing model parameters ...')
        best_acc = 0
        '''
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                c = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, 1.0 / c)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data = m.weight.data.zero_().add(1.0)
        '''


        best_acc2 = 0


        best_acc_ks2 = 0
        for m in model_ks2.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                c = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, 1.0 / c)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data = m.weight.data.zero_().add(1.0)

    else:
        print('==> Load pretrained model form', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['best_acc']
        model.load_state_dict(pretrained_model['state_dict'])

    if not args.cpu:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

        model2.cuda()
        model2 = torch.nn.DataParallel(model2, device_ids=range(torch.cuda.device_count()))

        model_ks2.cuda()
        model_ks2 = torch.nn.DataParallel(model_ks2, device_ids=range(torch.cuda.device_count()))
    print(model)

    # define solver and criterion
    base_lr = 0.1
    #full
    param_dict = dict(model.named_parameters())
    params = []
    for key, value in param_dict.items():
        params += [{'params':[value], 'lr': base_lr,
            'weight_decay':5e-4}]

        optimizer = optim.SGD(params, lr=base_lr, momentum=0.9, weight_decay=5e-4)


    #full2
    base_lr2 = 0.1
    param_dict2 = dict(model2.named_parameters())
    params2 = []
    for key, value in param_dict2.items():
        params2 += [{'params':[value], 'lr': base_lr2,
            'weight_decay':5e-4}]

        optimizer2 = optim.SGD(params2, lr=base_lr2, momentum=0.9, weight_decay=5e-4)


    #ks2
    base_lr2 = 0.01
    param_dict_ks2 = dict(model_ks2.named_parameters())
    params_ks2 = []
    for key, value in param_dict_ks2.items():
        params_ks2 += [{'params': [value], 'lr': base_lr2,
                    'weight_decay': 0.0001}]

        optimizer_ks2 = optim.Adam(params_ks2, lr=base_lr2, weight_decay=0.0001)




    criterion = nn.CrossEntropyLoss()



    # do the evaluation if specified
    if args.evaluate:
        test()
        exit(0)

    # start training



    for epoch in range(1, 270):
        adjust_learning_rate(optimizer, epoch)
        adjust_learning_rate(optimizer2, epoch)
        adjust_learning_rate(optimizer_ks2, epoch)



        train(epoch)


        acc_full = test()
        acc_full2 = test2()
        acc_ks2 = test_ks2()


























