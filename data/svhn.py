import os
import torch
import torchvision
from torchvision import transforms
import random
from torch.utils.data.sampler import SubsetRandomSampler
from args import args


class SVHN:
    def __init__(self, args):
        super(SVHN, self).__init__()

        data_root = os.path.join(args.data, "svhn")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        normalize = transforms.Normalize(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        )

        train_dataset = torchvision.datasets.SVHN(
            data_root,
            split='train',
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )

        if args.test_batch_size is None:
            args.test_batch_size = args.batch_size // 2

        test_dataset = torchvision.datasets.SVHN(
            data_root,
            split='test',
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )
        self.val_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs
        )

