import os
import pathlib
import random
import time
import shutil
import math

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F

from utils.conv_type import FixedSubnetConv, SampleSubnetConv
from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import (
    set_model_prune_rate,
    freeze_model_weights,
    freeze_model_subnet,
    save_checkpoint,
    get_lr,
    LabelSmoothing,
    init_model_weight_with_score,
)
from utils.schedulers import get_policy
import logging

from args import args
import importlib

import data
import models

from utils.builder import get_builder
from utils.eval_utils import accuracy

import tqdm


cifar_mean = (0.4914, 0.4822, 0.4465)
cifar_std = (0.2471, 0.2435, 0.2616)

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def validate_adv(val_loader, model, model_attack, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()
    model_attack.eval()

    if args.set == 'ImageNet':
        mean = imagenet_mean
        std = imagenet_std
    else:
        mean = cifar_mean
        std = cifar_std
    
    mu = torch.tensor(mean).view(3,1,1).cuda()
    std = torch.tensor(std).view(3,1,1).cuda()

    upper_limit = ((1 - mu)/ std)
    lower_limit = ((0 - mu)/ std)
    
    epsilon = (args.epsilon / 255.) / std
    # alpha = (args.alpha / 255.) / std
    alpha = (2 / 255.) / std
    
    end = time.time()
    for i, (X, y) in tqdm.tqdm(
        enumerate(val_loader), ascii=True, total=len(val_loader)
    ):

        X = X.cuda()
        y = y.cuda()

        pgd_delta = attack_pgd(model_attack, X, y, epsilon, alpha, lower_limit, upper_limit, attack_iters=20, restarts=1)
        # compute output
        output = model(X + pgd_delta)

        loss = criterion(output, y)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, y, topk=(1, 5))
        losses.update(loss.item(), X.size(0))
        top1.update(acc1.item(), X.size(0))
        top5.update(acc5.item(), X.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    progress.display(len(val_loader))

    return top1.avg, top5.avg


def attack_pgd(model, X, y, epsilon, alpha, lower_limit, upper_limit, attack_iters=20, restarts=1):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()

            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def main():
    # print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Simply call main_worker function
    main_worker(args)


def main_worker(args):
    # Set up directories
    args.task = 'transfer'

    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
    args.ckpt_base_dir = ckpt_base_dir

    log = logging.getLogger(__name__)
    log_path = os.path.join(run_base_dir, 'log.txt')
    handlers = [logging.FileHandler(log_path, mode='a+'),
                logging.StreamHandler()]
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=handlers)
    log.info(args)

    # pretrained models are saved at ./ckpt/ and named as modelname_prunerate_othercomment

    # create model and optimizer
    model_attack = get_model(args)
    set_model_prune_rate(model_attack, prune_rate=float(os.path.basename(args.pretrained).replace('.pth', '').split('_')[1]))
    
    model_attack = set_gpu(args, model_attack)

    pretrained(args.pretrained, model_attack)

    dirname = os.path.dirname(args.pretrained)
    ckpt_list = [os.path.join(dirname, path) for path in os.listdir(dirname)]
    ckpt_list.sort()

    data = get_dataset(args)
    criterion = nn.CrossEntropyLoss().cuda()
    
    acc_list = []
    for i, path in enumerate(ckpt_list):
        model = get_model(args)
        set_model_prune_rate(model, prune_rate=float(os.path.basename(path).replace('.pth', '').split('_')[1]))

        model = set_gpu(args, model)

        pretrained(path, model)

        acc1, acc5 = validate_adv(data.val_loader, model, model_attack, criterion, args)

        log.info('Robust Acc of %s: %f', path, acc1)
        acc_list.append(acc1)

        del model
    
    log.info('Acc list: %s', acc_list)

    log_dir_new = 'logs/log_'+args.name
    if not os.path.exists(log_dir_new):
        os.makedirs(log_dir_new)
    
    shutil.copyfile(log_path, os.path.join(log_dir_new, 'log_'+args.task+'.txt'))



def set_gpu(args, model):
    assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"

    model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    return model


def pretrained(path, model):
    if os.path.isfile(path):
        print("=> loading pretrained weights from '{}'".format(path))
        pretrained = torch.load(path)["state_dict"]
        model.load_state_dict(pretrained)

    else:
        print("=> no pretrained weights found at '{}'".format(path))
        exit()



def get_dataset(args):
    print(f"=> Getting {args.set} dataset")
    dataset = getattr(data, args.set)(args)

    return dataset


def get_model(args):
    if args.first_layer_dense:
        args.first_layer_type = "DenseConv"

    print("=> Creating model '{}'".format(args.arch))

    if args.set == 'ImageNet' or args.set == 'TinyImageNet':
        num_classes = 1000
    elif args.set == 'CIFAR100':
        num_classes = 100
    else:
        num_classes = 10

    model = models.__dict__[args.arch](num_classes=num_classes)

    return model


def _run_dir_exists(run_base_dir):
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    return log_base_dir.exists() or ckpt_base_dir.exists()


def get_directories(args):
    if args.config is None or args.name is None:
        raise ValueError("Must have name and config")

    config = pathlib.Path(args.config).stem
    if args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"runs/{config}/{args.name}/prune_rate={args.prune_rate}/{args.task}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{args.log_dir}/{config}/{args.name}/prune_rate={args.prune_rate}/{args.task}"
        )
    if args.width_mult != 1.0:
        run_base_dir = run_base_dir / "width_mult={}".format(str(args.width_mult))

    # if _run_dir_exists(run_base_dir):
    #     rep_count = 0
    #     while _run_dir_exists(run_base_dir / str(rep_count)):
    #         rep_count += 1

    #     run_base_dir = run_base_dir / str(rep_count)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    if not run_base_dir.exists():
        os.makedirs(run_base_dir)

    (run_base_dir / "settings.txt").write_text(str(args))

    return run_base_dir, ckpt_base_dir, log_base_dir




if __name__ == "__main__":
    main()
