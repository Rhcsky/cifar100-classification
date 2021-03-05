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
from utils import accuracy, margin_of_error

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
    max_len_exp = max([len(x) for x in model_path]) + 2

    print(f"|{'Model name':^{max_len_exp}}|{'Loss':^17}|{'Top1 err':^17}|{'Top1 acc':^17}|")

    except_list = []
    pl_mi = u"\u00B1"

    for path in model_path:
        checkpoint, args = None, None

        pth_file = path + "/model_best.pth"
        conf_file = path + "/params.json"

        params = json.load(open(conf_file))
        args.__dict__.update(params)

        checkpoint = torch.load(pth_file)

        if checkpoint is None or args is None:
            except_list.append(f"checkpoint and params are not exist in {path}")

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
        err1_list, val_loss_list = test(val_loader, model, criterion, path)

        err1, err1_moe = margin_of_error(err1_list)
        # err5, err5_moe = margin_of_error(err5_list)
        loss, loss_moe = margin_of_error(val_loss_list)

        err1_string = f'{err1:.3f} {pl_mi} {err1_moe:.3f}'
        # err5_string = f'{err5:.3f} {pl_mi} {err5_moe:.3f}'
        loss_string = f'{loss:.3f} {pl_mi} {loss_moe:.3f}'

        print(f"|{path:^{max_len_exp}}|{loss_string:^17}|{err1_string:^17}|{100 - err1}")


@torch.no_grad()
def test(val_loader, model, criterion, path):
    losses = []
    top1 = []
    # top5 = []

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

        losses.append(loss.item())
        top1.append(err1.item())
        # top5.append(err5.item())

    f.close()
    return top1, losses


if __name__ == "__main__":
    main()
