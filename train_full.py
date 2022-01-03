import os
import pathlib
import random
import time
import logging
import shutil

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

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


from args import args
import importlib

import data
import models

from utils.builder import get_builder


def main():
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Simply call main_worker function
    main_worker(args)


def main_worker(args):
    # Set up directories
    args.prune_rate = 1.0
    
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

    args.gpu = None
    train, validate, validate_adv, modifier = get_trainer(args)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model and optimizer
    model = get_model(args)
    model = set_gpu(args, model)

    if args.pretrained:
        pretrained(args, model)

    # freezing the subnet and set prune ratio to 1.0
    freeze_model_subnet(model)
    set_model_prune_rate(model, prune_rate=1.0)

    optimizer = get_optimizer(args, model)
    data = get_dataset(args)

    lr_policy = get_policy(args.lr_policy)(optimizer, args)

    if args.label_smoothing is None:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = LabelSmoothing(smoothing=args.label_smoothing)

    # optionally resume from a checkpoint
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    natural_acc1_at_best_robustness = None

    if args.resume:
        best_acc1, natural_acc1_at_best_robustness = resume(args, model, optimizer)

    # Data loading code
    if args.evaluate:
        if args.attack_type != 'None':
            acc1, acc5 = validate_adv(
                data.val_loader, model, criterion, args, writer=None, epoch=args.start_epoch
            )

            natural_acc1, natural_acc5 = validate(
                data.val_loader, model, criterion, args, writer=None, epoch=args.start_epoch
            )

            print('Natural Acc:', natural_acc1, 'Robust Acc:', acc1)
        
        else:
            acc1, acc5 = validate(
                data.val_loader, model, criterion, args, writer=None, epoch=args.start_epoch
            )

            print('Natural Acc:', acc1)

        return


    writer = SummaryWriter(log_dir=log_base_dir)
    epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
    validation_time = AverageMeter("validation_time", ":.4f", write_avg=False)
    train_time = AverageMeter("train_time", ":.4f", write_avg=False)
    progress_overall = ProgressMeter(
        1, [epoch_time, validation_time, train_time], prefix="Overall Timing"
    )

    end_epoch = time.time()
    args.start_epoch = args.start_epoch or 0
    acc1 = None

    # save_checkpoint(
    #     {
    #         "epoch": 0,
    #         "arch": args.arch,
    #         "state_dict": model.state_dict(),
    #         "best_acc1": best_acc1,
    #         "best_acc5": best_acc5,
    #         "best_train_acc1": best_train_acc1,
    #         "best_train_acc5": best_train_acc5,
    #         'natural_acc1_at_best_robustness': natural_acc1_at_best_robustness,
    #         "optimizer": optimizer.state_dict(),
    #         "curr_acc1": acc1 if acc1 else "Not evaluated",
    #     },
    #     False,
    #     filename=ckpt_base_dir / f"initial.state",
    #     save=False,
    # )

    # Start training
    for epoch in range(args.start_epoch, args.epochs):
        lr_policy(epoch, iteration=None)
        modifier(args, epoch, model)

        cur_lr = get_lr(optimizer)

        # train for one epoch
        start_train = time.time()
        train_acc1, train_acc5 = train(
            data.train_loader, model, criterion, optimizer, epoch, args, writer=writer, log=None
        )
        train_time.update((time.time() - start_train) / 60)

        # evaluate on validation set
        val_every = args.val_every if epoch > 60 else 10
        if epoch % val_every == 0 or epoch == args.epochs - 1:
            # evaluate on validation set
            start_validation = time.time()

            if args.attack_type != 'None':
                acc1, acc5 = validate_adv(data.val_loader, model, criterion, args, writer, epoch)
                natural_acc1, natural_acc5 = validate(data.val_loader, model, criterion, args, writer, epoch)
            else:
                acc1, acc5 = validate(data.val_loader, model, criterion, args, writer, epoch)

            validation_time.update((time.time() - start_validation) / 60)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            best_acc5 = max(acc5, best_acc5)
            best_train_acc1 = max(train_acc1, best_train_acc1)
            best_train_acc5 = max(train_acc5, best_train_acc5)

            if is_best and args.attack_type != 'None':
                natural_acc1_at_best_robustness = natural_acc1
                
            if is_best:
                print(f"==> New best, saving at {ckpt_base_dir / 'model_best.pth'}")

            if is_best or epoch == args.epochs - 1:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": args.arch,
                        "state_dict": model.state_dict(),
                        "best_acc1": best_acc1,
                        "best_acc5": best_acc5,
                        "best_train_acc1": best_train_acc1,
                        "best_train_acc5": best_train_acc5,
                        "natural_acc1_at_best_robustness": natural_acc1_at_best_robustness,
                        # "optimizer": optimizer.state_dict(),
                        "curr_acc1": acc1,
                        "curr_acc5": acc5,
                    },
                    is_best,
                    filename=ckpt_base_dir / f"epoch_{epoch}.state",
                    save=False,
                )

            if args.attack_type != 'None':
                log.info('Epoch[%d][%d] curr natural acc: %.2f, natural acc at best robustness: %.2f \n curr robust acc: %.2f, best robust acc: %.2f',
                args.epochs, epoch, natural_acc1, natural_acc1_at_best_robustness, acc1, best_acc1)
            else:
                log.info('Epoch[%d][%d] curr acc: %.2f, best acc: %.2f', args.epochs, epoch, acc1, best_acc1)

        # if args.conv_type == "SampleSubnetConv":
        #     count = 0
        #     sum_pr = 0.0
        #     for n, m in model.named_modules():
        #         if isinstance(m, SampleSubnetConv):
        #             # avg pr across 10 samples
        #             pr = 0.0
        #             for _ in range(10):
        #                 pr += (
        #                     (torch.rand_like(m.clamped_scores) >= m.clamped_scores)
        #                     .float()
        #                     .mean()
        #                     .item()
        #                 )
        #             pr /= 10.0
        #             writer.add_scalar("pr/{}".format(n), pr, epoch)
        #             sum_pr += pr
        #             count += 1

        #     args.prune_rate = sum_pr / count
        #     writer.add_scalar("pr/average", args.prune_rate, epoch)

        epoch_time.update((time.time() - end_epoch) / 60)
        progress_overall.display(epoch)
        progress_overall.write_to_tensorboard(
            writer, prefix="diagnostics", global_step=epoch
        )

        writer.add_scalar("test/lr", cur_lr, epoch)
        end_epoch = time.time()


    log_dir_new = 'logs/log_'+args.name
    if not os.path.exists(log_dir_new):
        os.makedirs(log_dir_new)
    
    shutil.copyfile(log_path, os.path.join(log_dir_new, 'log_'+args.task+'.txt'))


    # write_result_to_csv(
    #     best_acc1=best_acc1,
    #     best_acc5=best_acc5,
    #     best_train_acc1=best_train_acc1,
    #     best_train_acc5=best_train_acc5,
    #     prune_rate=args.prune_rate,
    #     curr_acc1=acc1,
    #     curr_acc5=acc5,
    #     base_config=args.config,
    #     name=args.name,
    # )


def get_trainer(args):
    print(f"=> Using trainer from trainers.{args.trainer}")
    trainer = importlib.import_module(f"trainers.{args.trainer}")

    if args.attack_type == 'None':
        return trainer.train, trainer.validate, None, trainer.modifier
    else:
        return trainer.train_adv, trainer.validate, trainer.validate_adv, trainer.modifier


def set_gpu(args, model):
    assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"

    model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    return model


def resume(args, model, optimizer):
    if os.path.isfile(args.resume):
        print(f"=> Loading checkpoint '{args.resume}'")

        checkpoint = torch.load(args.resume)
        if args.start_epoch is None:
            print(f"=> Setting new start epoch at {checkpoint['epoch']}")
            args.start_epoch = checkpoint["epoch"]

        best_acc1 = checkpoint["best_acc1"]
        natural_acc1_at_best_robustness = checkpoint["natural_acc1_at_best_robustness"]

        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

        return best_acc1, natural_acc1_at_best_robustness
    else:
        print(f"=> No checkpoint found at '{args.resume}'")


def pretrained(args, model):
    if os.path.isfile(args.pretrained):
        print("=> loading pretrained weights from '{}'".format(args.pretrained))
        pretrained = torch.load(args.pretrained)["state_dict"]

        model_state_dict = model.state_dict()
        for k, v in pretrained.items():
            if k not in model_state_dict or v.size() != model_state_dict[k].size():
                print("IGNORE:", k)
        pretrained = {
            k: v
            for k, v in pretrained.items()
            if (k in model_state_dict and v.size() == model_state_dict[k].size())
        }
        model_state_dict.update(pretrained)
        model.load_state_dict(model_state_dict)

    else:
        print("=> no pretrained weights found at '{}'".format(args.pretrained))
        exit()

    for n, m in model.named_modules():
        if isinstance(m, FixedSubnetConv):
            m.set_subnet()


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

    # applying sparsity to the network
    if (
        args.conv_type != "DenseConv"
        and args.conv_type != "SampleSubnetConv"
        and args.conv_type != "ContinuousSparseConv"
    ):
        if args.prune_rate < 0:
            raise ValueError("Need to set a positive prune rate")

        set_model_prune_rate(model, prune_rate=args.prune_rate)
        print(
            f"=> Rough estimate model params {sum(int(p.numel() * (1-args.prune_rate)) for n, p in model.named_parameters() if not n.endswith('scores'))}"
        )

    # freezing the weights if we are only doing subnet training
    # if args.freeze_weights:
    #     freeze_model_weights(model)

    return model


def get_optimizer(args, model):
    for n, v in model.named_parameters():
        if v.requires_grad:
            print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            print("<DEBUG> no gradient to", n)

    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0 if args.no_bn_decay else args.weight_decay,
                },
                {"params": rest_params, "weight_decay": args.weight_decay},
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )

    return optimizer


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


def write_result_to_csv(**kwargs):
    results = pathlib.Path("runs") / "results.csv"

    if not results.exists():
        results.write_text(
            "Date Finished, "
            "Base Config, "
            "Name, "
            "Prune Rate, "
            "Current Val Top 1, "
            "Current Val Top 5, "
            "Best Val Top 1, "
            "Best Val Top 5, "
            "Best Train Top 1, "
            "Best Train Top 5\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                "{base_config}, "
                "{name}, "
                "{prune_rate}, "
                "{curr_acc1:.02f}, "
                "{curr_acc5:.02f}, "
                "{best_acc1:.02f}, "
                "{best_acc5:.02f}, "
                "{best_train_acc1:.02f}, "
                "{best_train_acc5:.02f}\n"
            ).format(now=now, **kwargs)
        )


if __name__ == "__main__":
    main()
