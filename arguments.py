import argparse
import os
import json

parser = argparse.ArgumentParser(description="Cutmix PyTorch CIFAR-100")

parser.add_argument(
    "-j",
    "--workers",
    default=0,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--epochs", default=300, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "-b",
    "--batch_size",
    default=64,
    type=int,
    metavar="N",
    help="mini-batch size (default: 64)",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--weight-decay",
    "--wd",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
)
parser.add_argument(
    "--depth", default=20, type=int, help="depth of the network (default: 20)"
)
parser.add_argument(
    "--no-bottleneck",
    dest="bottleneck",
    action="store_false",
    help="to use basicblock for CIFAR datasets (default: bottleneck)",
)
parser.add_argument(
    "--verbose",
    dest="verbose",
    action="store_true",
    help="to print the status at every iteration",
)
parser.add_argument(
    "--alpha",
    default=300,
    type=float,
    help="number of new channel increases per depth (default: 300)",
)
parser.add_argument(
    "--expname", "-e", default="TEST", type=str, help="name of experiment"
)
parser.add_argument("--beta", default=1.0, type=float, help="hyperparameter beta")
parser.add_argument("--cutmix_prob", default=0.5, type=float, help="cutmix probability")
parser.add_argument("--resume", action="store_true", help="resume train")
parser.add_argument('-m', "--model", default="pyramid", help="Classification model")

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=False)
parser.set_defaults(resume=False)


def get_config():
    args = parser.parse_args()

    if args.resume:
        args = load_args(args)
    else:
        save_args(args)

    return args


def save_args(args):
    directory = f"runs/{args.expname}/"
    param_path = os.path.join(directory, "params.json")

    if not os.path.exists(f"runs/{args.expname}"):
        os.makedirs(directory)

    if not os.path.isfile(param_path):
        print(f"Save params in {param_path}")

        all_params = args.__dict__
        with open(param_path, "w") as fp:
            json.dump(all_params, fp, indent=4, sort_keys=True)
    else:
        print(f"Config file already exist.")
        raise ValueError


def load_args(args):
    param_path = os.path.join("runs/", args.expname, "params.json")
    params = json.load(open(param_path))

    args.__dict__.update(params)

    args.resume = True

    return args
