import numpy as np
import math

__all__ = ["multistep_lr", "cosine_lr", "constant_lr", 'cifar_piecewise', "get_policy"]


def get_policy(name):
    if name is None:
        return constant_lr

    out_dict = {
        "constant_lr": constant_lr,
        "cosine_lr": cosine_lr,
        "multistep_lr": multistep_lr,
        "multistep_lr_imagenet": multistep_lr_imagenet,
        "multistep_lr_imagenet_free": multistep_lr_imagenet_free,
        "cifar_piecewise": cifar_piecewise,
        "cifar_piecewise_adv": cifar_piecewise_adv,
        "ssfd":slow_start_fast_decay
    }

    return out_dict[name]


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def constant_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        else:
            lr = args.lr

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def cosine_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        else:
            e = epoch - args.warmup_length
            es = args.epochs - args.warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * args.lr

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def multistep_lr_imagenet(optimizer, args, **kwargs):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    def _lr_adjuster(epoch, iteration):
        lr = args.lr * (args.multistep_lr_gamma ** (epoch // args.multistep_lr_adjust))

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def multistep_lr_imagenet_free(optimizer, args, **kwargs):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    def _lr_adjuster(epoch, iteration):
        lr = args.lr * (args.multistep_lr_gamma ** (epoch // int(math.ceil(args.multistep_lr_adjust/args.n_repeats))))

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def multistep_lr(optimizer, args, **kwargs):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    def _lr_adjuster(epoch, iteration):
        if epoch < args.multistep[0]:
            lr = args.lr
        elif epoch >= args.multistep[0] and epoch < args.multistep[1]:
            lr = args.lr * 0.1
        elif epoch >= args.multistep[1]:
            lr = args.lr * 0.01

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def cifar_piecewise(optimizer, args, **kwargs):
    """Sets the learning rate to the initial LR decayed by 0.1 at 80-th and 120-th epochs"""

    def _lr_adjuster(epoch, iteration):
        if epoch < 80:
            lr = args.lr
        elif epoch >= 80 and epoch < 120:
            lr = args.lr * 0.1
        elif epoch >= 120:
            lr = args.lr * 0.01

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def cifar_piecewise_adv(optimizer, args, **kwargs):
    """Sets the learning rate to the initial LR decayed by 0.1 at 80-th and 120-th epochs"""

    def _lr_adjuster(epoch, iteration):
        if epoch < 50:
            lr = args.lr
        elif epoch >= 50 and epoch < 150:
            lr = args.lr * 0.1
        elif epoch >= 150:
            lr = args.lr * 0.01

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster



def _warmup_lr(base_lr, warmup_length, epoch):
    return base_lr * (epoch + 1) / warmup_length



def slow_start_fast_decay(optimizer, args, **kwargs):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    def _lr_adjuster(epoch, iteration):
        epoch = epoch % 10

        if epoch <= 4:
            lr = args.lr * (epoch+1)
        if epoch == 5:
            lr = args.lr * 2.5
        if epoch == 6:
            lr = args.lr * 1.25
        if epoch == 7:
            lr = args.lr * 0.625
        if epoch == 8:
            lr = args.lr * 0.3125
        if epoch == 9:
            lr = args.lr * 0.15625
            
        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster
