import argparse
import sys
import yaml

from config_rst import parser as _parser

args = None


def parse_arguments():
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

    # General Config
    parser.add_argument(
        "--data", help="path to dataset base directory", default="/mnt/disk1/datasets"
    )
    parser.add_argument("--optimizer", help="Which optimizer to use", default="sgd")
    parser.add_argument("--set", help="name of dataset", type=str, default="ImageNet")
    parser.add_argument(
        "-a", "--arch", metavar="ARCH", default="ResNet18", help="model architecture"
    )
    parser.add_argument(
        "--config", help="Config file to use (see configs dir)", default=None
    )
    parser.add_argument(
        "--log-dir", help="Where to save the runs. If None use ./runs", default=None
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=20,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 20)",
    )
    parser.add_argument(
        "--epochs",
        default=160,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch",
        default=None,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--test_batch_size",
        default=None,
        type=int,
        metavar="N",
        help="mini-batch size for test",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--warmup_length", default=0, type=int, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument("--num-classes", default=10, type=int)
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        default=None,
        type=str,
        help="use pre-trained model",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument(
        "--multigpu",
        default=None,
        type=lambda x: [int(a) for a in x.split(",")],
        help="Which GPUs to use for multigpu training",
    )

    # Learning Rate Policy Specific
    parser.add_argument(
        "--lr_policy", default="cifar_piecewise", help="Policy for the learning rate."
    )
    parser.add_argument(
        "--multistep_lr_adjust", default=30, type=int, help="Interval to drop lr"
    )
    parser.add_argument(
        "--multistep_lr_gamma", default=0.1, type=int, help="Multistep multiplier"
    )
    parser.add_argument(
        "--name", default=None, type=str, help="Experiment name to append to filepath"
    )
    parser.add_argument(
        "--save_every", default=-1, type=int, help="Save every ___ epochs"
    )
    parser.add_argument(
        "--prune-rate",
        default=0.0,
        help="Amount of pruning to do during sparse training",
        type=float,
    )
    parser.add_argument(
        "--low-data", default=1, help="Amount of data to use", type=float
    )
    parser.add_argument(
        "--width-mult",
        default=1.0,
        help="How much to vary the width of the network.",
        type=float,
    )
    parser.add_argument(
        "--nesterov",
        default=False,
        action="store_true",
        help="Whether or not to use nesterov for SGD",
    )
    parser.add_argument(
        "--random-subnet",
        action="store_true",
        help="Whether or not to use a random subnet when fine tuning for lottery experiments",
    )
    parser.add_argument(
        "--one-batch",
        action="store_true",
        help="One batch train set for debugging purposes (test overfitting)",
    )
    parser.add_argument(
        "--conv_type", type=str, default=None, help="What kind of sparsity to use"
    )
    parser.add_argument(
        "--freeze-weights",
        action="store_true",
        help="Whether or not to train only subnet (this freezes weights)",
    )
    parser.add_argument("--mode", default="fan_in", help="Weight initialization mode")
    parser.add_argument(
        "--nonlinearity", default="relu", help="Nonlinearity used by initialization"
    )
    parser.add_argument("--bn-type", default=None, help="BatchNorm type")
    parser.add_argument(
        "--init", default="kaiming_normal", help="Weight initialization modifications"
    )
    parser.add_argument(
        "--no-bn-decay", action="store_true", default=False, help="No batchnorm decay"
    )
    parser.add_argument(
        "--scale-fan", action="store_true", default=False, help="scale fan"
    )
    parser.add_argument(
        "--first-layer-dense", action="store_true", help="First layer dense or sparse"
    )
    parser.add_argument(
        "--last-layer-dense", action="store_true", help="Last layer dense or sparse"
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        help="Label smoothing to use, default 0.0",
        default=None,
    )
    parser.add_argument(
        "--first-layer-type", type=str, default=None, help="Conv type of first layer"
    )
    parser.add_argument(
        "--trainer", type=str, default="default", help="cs, ss, or standard training"
    )
    parser.add_argument(
        "--score-init-constant",
        type=float,
        default=None,
        help="Sample Baseline Subnet Init",
    )

    parser.add_argument(
        "--val_every", default=3, type=int, help="Validation every ___ epochs"
    )

    parser.add_argument('--attack_type', default='None', choices=['fgsm', 'fgsm-rs', 'pgd', 'free', 'None'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=10, type=float, help='Step size')
    parser.add_argument('--attack_iters', default=7, type=int, help='Attack iterations')
    
    parser.add_argument('--task', default='search', choices=['search', 'ft_inherit', 'ft_reinit', 'ft_full'])
    parser.add_argument(
        "--ft_init", default="kaiming_normal", help="Weight initialization for finetuning"
    )

    parser.add_argument(
        "--ft_full_mode", default='all', choices=['all','only_zero','decay_on_zero','low_lr_zero'], help="how to finetune the whole model"
    )
    
    parser.add_argument('--weight_decay_on_zero', default=1e-3, type=float, help='weight decay on the parameters intialized to be zero in the ft_full phase')

    parser.add_argument('--lr_scale_zero', default=1e-1, type=float, help='weight decay on the parameters intialized to be zero in the ft_full phase')

    parser.add_argument('--multistep', default=None, type=int, nargs='*',
                    help='lr switch point for multi step lr decay')

    parser.add_argument(
        "--automatic_resume", action="store_true", help="automatically resume"
    )

    parser.add_argument('--n_repeats', default=4, type=int, help='n_repeats in free adversarial training')

    parser.add_argument("--discard_mode", action="store_true", help="gradually discard lowest scores")
    parser.add_argument('--discard_epoch', default=10, type=int, help='discard the lowest score every x epoch')
    parser.add_argument('--discard_rate', default=0.1, type=float, help='discard x weight each time')

    parser.add_argument("--progressive_prune", action="store_true", help="progressively reduce the prune rate")

    parser.add_argument("--once_for_all_list", default=None, type=float, nargs='*',
                         help="specify the target prune rate")

    parser.add_argument("--not_strict", action="store_true", help="if not use strict when loading model")

    parser.add_argument(
        "--pretrained2",
        default=None,
        type=str,
        help="the second pre-trained model dir for calculating cosine similarity",
    )

    args = parser.parse_args()

    # Allow for use from notebook without config file
    if len(sys.argv) > 1:
        get_config(args)

    return args


def get_config(args):
    # get commands from command line
    override_args = _parser.argv_to_vars(sys.argv)

    # load yaml file
    yaml_txt = open(args.config).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {args.config}")
    args.__dict__.update(loaded_yaml)


def run_args():
    global args
    if args is None:
        args = parse_arguments()


run_args()
