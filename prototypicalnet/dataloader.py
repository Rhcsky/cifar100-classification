from torch.utils.data import DataLoader
from torchvision import datasets as dset
import torch
import numpy as np
import warnings

from torchvision import transforms

warnings.filterwarnings("ignore")


def get_dataloader(args):
    print("Loading data...", end='')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_dataset = dset.CIFAR100('../data', train=True, download=True,
                                  transform=transform)
    test_dataset = dset.CIFAR100('../data', train=False, download=False,
                                 transform=transform)

    labels = []
    for data in train_dataset:
        labels.append(data[1])

    sampler_tr = PrototypicalBatchSampler(torch.LongTensor(labels), args.classes_per_it_tr, args.num_support_tr,
                                          args.num_query_tr,
                                          args.iterations)

    labels = []
    for data in test_dataset:
        labels.append(data[1])
    sampler_val = PrototypicalBatchSampler(torch.LongTensor(labels), args.classes_per_it_val, args.num_support_val,
                                           args.num_query_val,
                                           args.iterations)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler_tr)
    test_loader = DataLoader(test_dataset, batch_sampler=sampler_val)

    print("done")

    return train_loader, test_loader


class PrototypicalBatchSampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, labels, classes_per_it, num_samples_support, num_samples_query, iterations):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        '''
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.num_samples_support = num_samples_support
        self.num_samples_query = num_samples_query
        self.iterations = iterations

        self.classes, self.counts = torch.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self.labels))
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] += 1

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        nss = self.num_samples_support
        nsq = self.num_samples_query
        cpi = self.classes_per_it

        for it in range(self.iterations):
            batch_s = torch.LongTensor(nss * cpi)
            batch_q = torch.LongTensor(nsq * cpi)
            c_idxs = torch.randperm(len(self.classes))[:cpi]  # 랜덤으로 클래스 60개 선택
            for i, c in enumerate(self.classes[c_idxs]):
                s_s = slice(i * nss, (i + 1) * nss)  # 하나의 클래스당 선택한 support 이미지
                s_q = slice(i * nsq, (i + 1) * nsq)  # 하나의 클래스당 선택한 query 이미지
                # FIXME when torch.argwhere will exists
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:nss + nsq]
                batch_s[s_s] = self.indexes[label_idx][sample_idxs][:nss]
                batch_q[s_q] = self.indexes[label_idx][sample_idxs][nss:]
            batch = torch.cat((batch_s, batch_q))
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations
