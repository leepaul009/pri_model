import argparse
import inspect
import json
import math
import multiprocessing
import os
import pickle
import random
import subprocess
import sys
import time
import pdb
from collections import defaultdict
from multiprocessing import Process
from random import randint
from typing import Dict, List, Tuple, NamedTuple, Any, Union, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.path import Path
from matplotlib.pyplot import MultipleLocator
from torch import Tensor


def add_argument(parser):
    assert isinstance(parser, argparse.ArgumentParser)
    # Required parameters
    parser.add_argument("--use_cpu",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("-e", "--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true')

    parser.add_argument("--data_name",
                        default='pri_data',
                        type=str)
    
    parser.add_argument("--label_bin",
                        action='store_true',
                        help="Whether to run label_bin.")

    parser.add_argument("--data_dir",
                        default='data/pri_data/train/',
                        type=str)
    parser.add_argument("--data_dir_for_val",
                        default='data/pri_data/val/',
                        type=str)
    parser.add_argument("--data_dir_for_test",
                        default='data/pri_data/test/',
                        type=str)
    parser.add_argument("--output_dir", default="tmp/", type=str)
    parser.add_argument("--log_dir", default=None, type=str)
    parser.add_argument("--temp_file_dir", default=None, type=str)

    parser.add_argument("--resume",
                        action='store_true',
                        help="Whether to run resume.")
    parser.add_argument("--resume_path",
                        default='',
                        type=str)
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--test_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for test.")
    parser.add_argument("--model_recover_path",
                        default=None,
                        type=str)
    parser.add_argument("--learning_rate", default=0.001, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="The weight decay rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=100.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--hidden_size",
                        default=128,
                        type=int)
    parser.add_argument("--hidden_dropout_prob",
                        default=0.1,
                        type=float)
    parser.add_argument("--sub_graph_depth",
                        default=3,
                        type=int)
    parser.add_argument("--global_graph_depth",
                        default=1,
                        type=int)
    parser.add_argument("--debug",
                        action='store_true')
    parser.add_argument("--initializer_range",
                        default=0.02,
                        type=float)
    parser.add_argument("--sub_graph_batch_size",
                        default=8000,
                        type=int) # useless
    parser.add_argument("-d", "--distributed_training",
                        nargs='?',
                        default=8,
                        const=4,
                        type=int)
    parser.add_argument("--cuda_visible_device_num",
                        default=None,
                        type=int)
    parser.add_argument("--use_map",
                        action='store_true')
    parser.add_argument("--reuse_temp_file",
                        action='store_true')
    parser.add_argument("--old_version",
                        action='store_true')
    parser.add_argument("--max_distance",
                        default=50.0,
                        type=float)
    parser.add_argument("--no_sub_graph",
                        action='store_true')
    parser.add_argument("--no_agents",
                        action='store_true')
    parser.add_argument("--other_params",
                        nargs='*',
                        default=[],
                        type=str)
    parser.add_argument("-ep", "--eval_params",
                        nargs='*',
                        default=[],
                        type=str)
    parser.add_argument("-tp", "--train_params",
                        nargs='*',
                        default=[],
                        type=str)
    parser.add_argument("--not_use_api",
                        action='store_true')
    parser.add_argument("--core_num",
                        default=1,
                        type=int)
    parser.add_argument("--visualize",
                        action='store_true')
    parser.add_argument("--train_extra",
                        action='store_true')
    parser.add_argument("--use_centerline",
                        action='store_true')
    parser.add_argument("--autoregression",
                        nargs='?',
                        default=None,
                        const=2,
                        type=int)
    parser.add_argument("--lstm",
                        action='store_true')
    parser.add_argument("--add_prefix",
                        default=None)
    parser.add_argument("--attention_decay",
                        action='store_true')
    parser.add_argument("--placeholder",
                        default=0.0,
                        type=float)
    parser.add_argument("--multi",
                        nargs='?',
                        default=None,
                        const=6,
                        type=int)
    parser.add_argument("--method_span",
                        nargs='*',
                        default=[0, 1],
                        type=int)
    parser.add_argument("--nms_threshold",
                        default=None,
                        type=float)
    parser.add_argument("--stage_one_K", type=int)
    parser.add_argument("--master_port", default='12355')
    parser.add_argument("--gpu_split",
                        nargs='?',
                        default=0,
                        const=2,
                        type=int)
    parser.add_argument("--waymo",
                        action='store_true')
    parser.add_argument("--argoverse",
                        action='store_true')
    parser.add_argument("--nuscenes",
                        action='store_true')
    parser.add_argument("--future_frame_num",
                        default=80,
                        type=int)
    parser.add_argument("--future_test_frame_num",
                        default=16,
                        type=int)
    parser.add_argument("--single_agent",
                        action='store_true',
                        default=True)
    parser.add_argument("--agent_type",
                        default=None,
                        type=str)
    parser.add_argument("--inter_agent_types",
                        default=None,
                        nargs=2,
                        type=str)
    parser.add_argument("--mode_num",
                        default=6,
                        type=int)
    parser.add_argument("--local_rank", type=int, default=0)

    ## input features
    parser.add_argument("--pwm_type",
                        default='',
                        type=str,
                        help='use pwm feature as input, and choose pwm type')
    parser.add_argument("--use_chemistry",
                        action='store_true',
                        help='use chemistry feature as input')
    parser.add_argument("--use_prot_chm_feature",
                        action='store_true',
                        help='use chemistry feature as input')
    parser.add_argument("--use_repeat_sampler",
                        action='store_true',
                        help='use chemistry feature as input')
    parser.add_argument("--display_steps",
                        default=10,
                        type=int)
    parser.add_argument("--step_lr",
                        action='store_true',
                        help='set to update lr by steps')
    parser.add_argument("--steps_update_lr",
                        default=2000,
                        type=int,
                        help='update lr by setting steps')   

class Args:
    data_dir = None
    data_dir_for_val = None
    data_dir_for_test = None
    data_name = None
    data_kind = None
    
    label_bin = None
    
    use_cpu = None
    do_train = None
    do_eval = None
    do_test = None

    train_batch_size = None
    eval_batch_size = None
    test_batch_size = None

    debug = None
    seed = None

    distributed_training = None
    cuda_visible_device_num = None
    log_dir = None
    learning_rate = None
    
    hidden_size = None
    sub_graph_depth = None
    global_graph_depth = None
    train_batch_size = None
    num_train_epochs = None
    initializer_range = None
    sub_graph_batch_size = None
    temp_file_dir = None
    output_dir = None
    use_map = None
    reuse_temp_file = None
    old_version = None
    model_recover_path = None
    
    resume = None
    resume_path = None
    max_distance = None
    no_sub_graph = None
    other_params: Dict = None
    eval_params = None
    train_params = None
    no_agents = None
    not_use_api = None
    core_num = None
    visualize = None
    train_extra = None
    hidden_dropout_prob = None
    use_centerline = None
    autoregression = None
    lstm = None
    add_prefix = None
    attention_decay = None
    
    placeholder = None
    multi = None
    method_span = None
    waymo = None
    argoverse = None
    nuscenes = None
    single_agent = None
    agent_type = None
    future_frame_num = None
    no_cuda = None
    mode_num = None
    nms_threshold = None
    inter_agent_types = None
    local_rank = None
    ## input features
    pwm_type = None
    use_chemistry = None
    use_prot_chm_feature = None
    use_repeat_sampler = None
    display_steps = None
    step_lr = None
    steps_update_lr = None

args: Args = None

logger = None

def get_time():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

time_begin = get_time()

def init(args_: Args, logger_):
    global args, logger
    args = args_
    logger = logger_

    if not args.do_eval and not args.debug and os.path.exists(args.output_dir):
        print('{} {} exists'.format(get_color_text('Warning!'), args.output_dir))
        input()

    if args.do_eval:
        assert os.path.exists(args.output_dir)
        assert os.path.exists(args.data_dir_for_val)
    else:
        assert os.path.exists(args.data_dir)

    if args.log_dir is None:
        args.log_dir = args.output_dir
    if args.temp_file_dir is None:
        args.temp_file_dir = os.path.join(args.output_dir, 'temp_file')
    else:
        args.reuse_temp_file = True
        args.temp_file_dir = os.path.join(args.temp_file_dir, 'temp_file')

    dic = {}
    for i, param in enumerate(args.other_params + args.eval_params + args.train_params):
        if '=' in param:
            index = str(param).index('=')
            key = param[:index]
            value = param[index + 1:]
            # key, value = param.split('=')
            dic[key] = value if not str(value).isdigit() else int(value)
        else:
            dic[param] = True
    args.other_params = dic

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.temp_file_dir, exist_ok=True)
    if not args.do_eval and not args.debug:
        src_dir = os.path.join(args.output_dir, 'src')
        if os.path.exists(src_dir):
            subprocess.check_output('rm -r {}'.format(src_dir), shell=True, encoding='utf-8')
        os.makedirs(src_dir, exist_ok=False)
        for each in os.listdir('src'):
            is_dir = '-r' if os.path.isdir(os.path.join('src', each)) else ''
            subprocess.check_output(f'cp {is_dir} {os.path.join("src", each)} {src_dir}', shell=True, encoding='utf-8')
        with open(os.path.join(src_dir, 'cmd'), 'w') as file:
            file.write(' '.join(sys.argv))
    args.model_save_dir = os.path.join(args.output_dir, 'model_save')
    os.makedirs(args.model_save_dir, exist_ok=True)

    def init_args_do_eval():
        if args.argoverse:
            args.data_dir = args.data_dir_for_val if not args.do_test else 'test_obs/data/'
        if args.model_recover_path is None:
            args.model_recover_path = os.path.join(args.output_dir, 'model_save', 'model.16.bin')
        elif len(args.model_recover_path) <= 2:
            args.model_recover_path = os.path.join(args.output_dir, 'model_save',
                                                   'model.{}.bin'.format(args.model_recover_path))
        args.do_train = False

        if 'set_predict' in args.other_params:
            # TODO
            pass

        if len(args.method_span) != 2:
            args.method_span = [args.method_span[0], args.method_span[0] + 1]

        if args.mode_num != 6:
            add_eval_param(f'mode_num={args.mode_num}')

    def init_args_do_train():
        # if 'interactive' in args.other_params:
        #     args.data_dir = 'tf_example/validation_interactive/'
        pass

    if args.do_eval:
        init_args_do_eval()
    else:
        init_args_do_train()

    print(dict(sorted(vars(args_).items())))
    # print(json.dumps(vars(args_), indent=4))
    args_dict = vars(args)
    print()
    logger.info("***** args *****")
    for each in ['output_dir', 'other_params']:
        if each in args_dict:
            temp = args_dict[each]
            if each == 'other_params':
                temp = [param if args.other_params[param] is True else (param, args.other_params[param]) for param in
                        args.other_params]
            print("\033[31m" + each + "\033[0m", temp)
    logging(vars(args_), type='args', is_json=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(os.path.join(args.temp_file_dir, time_begin), exist_ok=True)

    if isinstance(args.data_dir, str):
        args.data_dir = [args.data_dir]

    assert args.do_train or args.do_eval

def get_color_text(text, color='red'):
    if color == 'red':
        return "\033[31m" + text + "\033[0m"
    else:
        assert False

def add_eval_param(param):
    if param not in args.eval_params:
        args.eval_params.append(param)

def get_name(name='', append_time=False):
    if name.endswith(time_begin):
        return name
    prefix = 'test.' if args.do_test else 'eval.' if args.do_eval and not args.do_train else ''
    prefix = 'debug.' + prefix if args.debug else prefix
    prefix = args.add_prefix + '.' + prefix if args.add_prefix is not None else prefix
    suffix = '.' + time_begin if append_time else ''
    return prefix + str(name) + suffix

files_written = {}

def logging(*inputs, prob=1.0, type='1', is_json=False, affi=True, sep=' ', to_screen=False, append_time=False, as_pickle=False):
    """
    Print args into log file in a convenient style.
    """
    if to_screen:
        print(*inputs, sep=sep)
    if not random.random() <= prob or not hasattr(args, 'log_dir'):
        return

    file = os.path.join(args.log_dir, get_name(type, append_time))
    if as_pickle:
        with open(file, 'wb') as pickle_file:
            assert len(inputs) == 1
            pickle.dump(*inputs, pickle_file)
        return
    if file not in files_written:
        with open(file, "w", encoding='utf-8') as fout:
            files_written[file] = 1
    inputs = list(inputs)
    the_tensor = None
    for i, each in enumerate(inputs):
        if isinstance(each, torch.Tensor):
            # torch.Tensor(a), a must be Float tensor
            if each.is_cuda:
                each = each.cpu()
            inputs[i] = each.data.numpy()
            the_tensor = inputs[i]
    np.set_printoptions(threshold=np.inf)

    with open(file, "a", encoding='utf-8') as fout:
        if is_json:
            for each in inputs:
                print(json.dumps(each, indent=4), file=fout)
        elif affi:
            print(*tuple(inputs), file=fout, sep=sep)
            if the_tensor is not None:
                print(json.dumps(the_tensor.tolist()), file=fout)
            print(file=fout)
        else:
            print(*tuple(inputs), file=fout, sep=sep)
            print(file=fout)

def de_merge_tensors(tensor: Tensor, lengths):
    return [tensor[i, :lengths[i]] for i in range(len(lengths))]

def merge_tensors(tensors: List[torch.Tensor], device, hidden_size=None) -> Tuple[Tensor, List[int]]:
    """
    merge a list of tensors into a tensor
    """
    lengths = []
    # hidden_size = args.hidden_size if hidden_size is None else hidden_size
    if hidden_size is None:
        hidden_size = tensors[0].shape[-1]
    for tensor in tensors:
        lengths.append(tensor.shape[0] if tensor is not None else 0)
    res = torch.zeros([len(tensors), max(lengths), hidden_size], device=device)
    for i, tensor in enumerate(tensors):
        if tensor is not None:
            res[i][:tensor.shape[0]] = tensor
    return res, lengths

def batch_list_to_batch_tensors(batch):
    return [each for each in batch]

def get_from_mapping(mapping: List[Dict], key=None):
    # if key is None:
    #     line_context = inspect.getframeinfo(inspect.currentframe().f_back).code_context[0]
    #     key = line_context.split('=')[0].strip()
    return [each[key] for each in mapping]




