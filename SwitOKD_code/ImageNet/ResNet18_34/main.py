import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.nn.functional as F


from models import resnet18, resnet34

from folder2lmdb import ImageFolderLMDB

# set the seed
torch.manual_seed(1)
torch.cuda.manual_seed(1)

import sys
import gc
cwd = os.getcwd()
sys.path.append(cwd+'/../')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    help='model architecture (default: alexnet)')

parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.90, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--print-freq', '-p', default=1000, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--resume', default='./', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    default=False, help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')

best_prec1_T = 0
best_prec1_S = 0


def main():
    global args, best_prec1_T, best_prec1_S
    args = parser.parse_args()

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    model_T = resnet34.resnet34()
    model_S = resnet18.resnet18()


    model_T.cuda()
    model_T = torch.nn.DataParallel(model_T, device_ids=range(torch.cuda.device_count()))

    model_S.cuda()
    model_S = torch.nn.DataParallel(model_S, device_ids=range(torch.cuda.device_count()))
    print(model_S)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer_T = torch.optim.SGD(model_T.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    optimizer_S = torch.optim.SGD(model_S.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)


    # for m in model.modules():
    #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #         c = float(m.weight.data[0].nelement())
    #         m.weight.data = m.weight.data.normal_(0, 1.0/c)
    #     elif isinstance(m, nn.BatchNorm2d):
    #         m.weight.data = m.weight.data.zero_().add(1.0)

    # optionally resume from a checkpoint
    if args.resume:
        Teacher_checkpoint = args.resume + 'Teacher_checkpoint.pth.tar'
        if os.path.isfile(Teacher_checkpoint):
            print("=> loading Teacher_checkpoint '{}'".format(Teacher_checkpoint))
            checkpoint = torch.load(Teacher_checkpoint)
            args.start_epoch = checkpoint['epoch']
            #best_prec1_T = checkpoint['best_prec1_T']
            best_prec1_T = 0
            model_T.load_state_dict(checkpoint['state_dict'])
            optimizer_T.load_state_dict(checkpoint['optimizer_T'])
            print("=> loaded Teacher_checkpoint '{}' (epoch {})"
                  .format(Teacher_checkpoint, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(Teacher_checkpoint))


        Student_checkpoint = args.resume + 'Student_checkpoint.pth.tar'
        if os.path.isfile(Student_checkpoint):
            print("=> loading Student_checkpoint '{}'".format(Student_checkpoint))
            checkpoint = torch.load(Student_checkpoint)
            args.start_epoch = checkpoint['epoch']
            #best_prec1_S = checkpoint['best_prec1_S']
            best_prec1_S = 0
            model_S.load_state_dict(checkpoint['state_dict'])
            optimizer_S.load_state_dict(checkpoint['optimizer_S'])
            print("=> loaded Student_checkpoint '{}' (epoch {})"
                  .format(Student_checkpoint, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(Student_checkpoint))

    cudnn.benchmark = True


    # Data loading code (lmdb)
    traindir = os.path.join('/qianbiao/dataset/Imagenet', 'ILSVRC2012_img_train.lmdb')
    valdir = os.path.join('/qianbiao/dataset/Imagenet', 'ILSVRC2012_img_val.lmdb')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = ImageFolderLMDB(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)


    val_dataset = ImageFolderLMDB(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)





    for epoch in range(args.start_epoch, args.epochs):
      
        adjust_learning_rate(optimizer_T, epoch)
        adjust_learning_rate(optimizer_S, epoch)
     
        # train for one epoch
        train(train_loader, model_T, model_S, criterion, optimizer_T, optimizer_S, epoch)

        # evaluate on validation set
        prec1_T = validate_T(val_loader, model_T, criterion, epoch)
        prec1_S = validate_S(val_loader, model_S, criterion, epoch)




        # T
        # remember best prec@1 and save checkpoint
        is_best = prec1_T > best_prec1_T
        best_prec1_T = max(prec1_T, best_prec1_T)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model_T.state_dict(),
            'best_prec1_T': model_T,
            'optimizer_T' : optimizer_T.state_dict(),
        }, is_best, flag='Teacher')

        # S
        # remember best prec@1 and save checkpoint
        is_best_S = prec1_S > best_prec1_S
        best_prec1_S = max(prec1_S, best_prec1_S)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model_S.state_dict(),
            'best_prec1_S': best_prec1_S,
            'optimizer_S': optimizer_S.state_dict(),
        }, is_best_S, flag='Student')
       




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


def train(train_loader, model_T, model_S, criterion, optimizer_T, optimizer_S, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model_T.train()
    model_S.train()




    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target_onehot = torch.autograd.Variable((torch.zeros(input.size()[0], 1000).cuda()).scatter_(1, target.view(target.size()[0], 1).cuda(), 1))
        #print(target_onehot.size())

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)


        
        # compute output
        optimizer_T.zero_grad()
        optimizer_S.zero_grad()

        output_T = model_T(input_var)
        output_S = model_S(input_var)


        s_label = dist_s_label(target_onehot, output_S.detach())
        t_label = dist_s_label(target_onehot, output_T.detach())

        #norm
        ps_pt = dist_s_t(output_T.detach(), output_S.detach(), 1)

        epsilon = torch.exp(-1 * t_label / (s_label + t_label))
        delta = s_label - epsilon * t_label




        if ps_pt > delta and t_label < s_label:

            loss_S = criterion(output_S, target_var) + \
                     kl_div(output_T.detach(), output_S, 1)


            loss_S.backward()

            optimizer_S.step()

        else:

            loss_T = criterion(output_T, target_var) + \
                     kl_div(output_S.detach(), output_T, 1)

            loss_S = criterion(output_S, target_var) + \
                     kl_div(output_T.detach(), output_S, 1)


            loss_T.backward()
            loss_S.backward()

            optimizer_T.step()
            optimizer_S.step()





        # measure accuracy and record loss
        prec1, prec5 = accuracy(output_S.data, target, topk=(1, 5))
        losses.update(loss_S.data, input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


        gc.collect()


def validate_T(val_loader, model_T, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model_T.eval()

    end = time.time()

    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model_T(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

            print('epoch: ' + str(epoch) + ' validate_T:' + ' T:entropy ' + str(compute_entropy(output, 1).item()))


    print('Teacher: * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def validate_S(val_loader, model_S, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model_S.eval()

    end = time.time()

    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model_S(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))


            print('epoch: ' + str(epoch) + ' validate_S:' + ' S:entropy ' + str(compute_entropy(output, 1).item()))



    print('Student: * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='_checkpoint.pth.tar', flag=''):
    path = './'
    torch.save(state, path+flag+filename)
    if is_best:
        shutil.copyfile(path+flag+filename, path+flag+'_model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 25 epochs"""
    epoch_list = [30, 60, 90]
    

    if epoch in epoch_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1


    for param_group in optimizer.param_groups:
        print ('Learning rate:', param_group['lr'])

    


    


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
