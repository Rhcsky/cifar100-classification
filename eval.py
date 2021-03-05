import argparse
import os
from glob import glob
import json

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from model.resnet import ResNet
from model.wide_resnet import get_wide_resnet
from model.pyramidnet import PyramidNet
from bsconv.replacers import BSConvS_Replacer
from dataloader import get_dataloader
from utils import accuracy, AverageMeter

parser = argparse.ArgumentParser(description="Cutmix PyTorch CIFAR-100 Test")
parser.add_argument("--model_path", default="./runs/*", type=str, metavar="PATH")
parser.add_argument("--batch_size", default=64)
parser.add_argument("--workers", default=0)

best_err1 = 100
best_err5 = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
margin = 12

def main():
    global args, best_err1, best_err5
    args = parser.parse_args()

    train_loader, val_loader = get_dataloader(args)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    model_path = glob(args.model_path)
    max_len_exp = max([len(os.path.basename(x)) for x in model_path]) + 2

    print(f"|{'Model name':^{max_len_exp}}|{'Loss':^{margin}}|{'Top1 err':^{margin}}|{'Top1 acc':^{margin}}|")

    except_list = []
    pl_mi = u"\u00B1"

    for path in model_path:
        pth_file = path + "/model_best.pth"
        conf_file = path + "/params.json"

        params = json.load(open(conf_file))
        args.__dict__.update(params)

        checkpoint = torch.load(pth_file)

        if args.model == 'resnet':
            model = ResNet(args.depth, 100, args.bottleneck)
        elif args.model.startswith('wrn'):
            model = get_wide_resnet(architecture='wrn28_3.26_bsconvs_p1d4', num_classes=100)
        elif args.model.startswith('pyramid'):
            model = PyramidNet(depth=32, alpha=295, bottleneck=True)

            p_frac = [1, 4]
            p = p_frac[0] / p_frac[1]
            replacer = BSConvS_Replacer(p=p, with_bn=True)
            model = replacer.apply(model)
        else:
            raise ValueError

        model.to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # evaluate on validation set
        err1, loss = test(val_loader, model, criterion, path)

        print(f"|{os.path.basename(path):^{max_len_exp}}|{loss:^{margin}.3f}|{err1:^{margin}}|{100 - err1:^{margin}.3f}|")


@torch.no_grad()
def test(val_loader, model, criterion, path):
    losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter

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
        # top5.append(err5.item())

    f.close()
    return top1.avg, losses.avg


if __name__ == "__main__":
    main()
