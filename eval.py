import argparse
import os
from glob import glob

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from model.resnet import ResNet
from dataloader import get_dataloader
from utils import AverageMeter, accuracy

parser = argparse.ArgumentParser(description="Cutmix PyTorch CIFAR-100 Test")
parser.add_argument("--model_path", default="./runs/*", type=str, metavar="PATH")
parser.add_argument("--batch_size", default=64)
parser.add_argument("--workers", default=0)

best_err1 = 100
best_err5 = 100
device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    global args, best_err1, best_err5
    args = parser.parse_args()

    train_loader, val_loader = get_dataloader(args)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    model_path = glob(args.model_path)

    print(f"|{'Model name':^20}|{'Top1 err':^15}|{'Top5 err':^15}|{'Top1 acc':^15}|")
    for path in model_path:
        if os.path.basename(path).startswith('1'):
            continue
        pth_file = path + "/model_best.pth"
        # conf_file = path + "/params.json"

        # params = json.load(open(conf_file))
        # args.__dict__.update(params)

        checkpoint = torch.load(pth_file)

        if args.model == 'resnet':
            model = ResNet(args.depth, 100, args.bottleneck)
        elif args.model.startswith('wrn'):
            model = get_wide_resnet(architecture=args.model, num_classes=100)
        else:
            model = get_wide_resnet()

        model.to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # evaluate on validation set
        err1, err5, val_loss = validate(val_loader, model, criterion, path)

        print(f"|{os.path.basename(path):^20}|{err1:^15}|{err5:^15}|{100 - err1:^15}|")


def validate(val_loader, model, criterion, path):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    f = open(f'{path + "/res"}.txt', "w")

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        target = target.to(device)

        output = model(input.to(device))
        loss = criterion(output, target)

        # measure accuracy and record loss
        (err1, err5), pred_list = accuracy(output.data, target, topk=(1, 5))

        for pred in pred_list:
            f.write(str(pred.item()) + "\n")

        losses.update(loss.item(), input.size(0))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

    f.close()
    return top1.avg, top5.avg, losses.avg


if __name__ == "__main__":
    main()
