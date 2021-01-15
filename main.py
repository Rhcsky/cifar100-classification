from configuration import get_config
from dataloader import get_dataloader

args = get_config()
best_err1 = 100


def run():
    global args, best_err1

    train_loader, test_loader = get_dataloader(args)


def train():
    pass


def test():
    pass
