import os
from glob import glob

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from configuration import get_config
from dataloader import get_dataloader
from classificator.model.resnet import ResNet
from classificator.model.wide_resnet import get_wide_resnet
from utils import AverageMeter, accuracy, optional_weight_decay_param

best_err1 = 100
best_err5 = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    global args, best_err1, best_err5, device
    args = get_config()

    train_loader, val_loader = get_dataloader(args)

    if args.model == 'resnet':
        model = ResNet(args.depth, 100, args.bottleneck)
    elif args.model.startswith('wrn'):
        model = get_wide_resnet(architecture=args.model, num_classes=100)
    else:
        model = get_wide_resnet()

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(optional_weight_decay_param(model, args.weight_decay), args.lr, momentum=args.momentum,
                                nesterov=True)

    writer = SummaryWriter(f'runs/{args.expname}')
    cudnn.benchmark = True

    if args.resume:
        checkpoint = torch.load(sorted(glob(f'runs/{args.expname}/checkpoint_*.pth'), key=len)[-1])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_err1 = checkpoint['best_err1']
        best_err5 = checkpoint['best_err5']

        print(f"load exp {args.expname}")
    else:
        start_epoch = 0

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200, 250], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(train_loader),
    #                                                 epochs=args.epochs - start_epoch, pct_start=0.1)
    print(f"model parameter : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    for epoch in range(start_epoch, args.epochs):

        # adjust_learning_rate(optimizer, epoch)

        train_loss = train(train_loader, model, optimizer, criterion)
        err1, err5, val_loss = validate(val_loader, model, criterion)

        if err1 <= best_err1:
            is_best = True
            best_err1, best_err5 = err1, err5
        else:
            is_best = False

        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_err1': best_err1,
            'best_err5': best_err5,
            'optimizer_state_dict': optimizer.state_dict(),
        }, is_best)
        
        writer.add_scalar("Loss/Train", train_loss)
        writer.add_scalar("Loss/Val", val_loss)
        writer.add_scalar("Err/Top1", err1)
        writer.add_scalar("Err/Top5", err5)

        print(f"[{epoch}/{args.epochs}] {train_loss:.3f}, {val_loss:.3f}, {err1}, {err5}, # {best_err1}, {best_err5}")

        scheduler.step()

    writer.close()


def train(train_loader, model, optimizer, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    for i, data in enumerate(train_loader):
        input, target = data[0].to(device), data[1].to(device)

        r = np.random.rand(1)
        if args.beta > 0 and r < args.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            # compute output
            output = model(input)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        else:
            # compute output
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        (err1, err5), _ = accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, data in enumerate(val_loader):
        input, target = data[0].to(device), data[1].to(device)

        output = model(input.cuda())
        loss = criterion(output, target)

        # measure accuracy and record loss
        (err1, err5), _ = accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best):
    directory = f"runs/{args.expname}/"
    filename = directory + f"checkpoint_{state['epoch']}.pth"

    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(state, filename)

    if is_best:
        filename = directory + "model_best.pth"
        torch.save(state, filename)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
