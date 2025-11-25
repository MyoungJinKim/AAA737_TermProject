"""
Adapted from salesforce@LAVIS. Below is the original copyright:
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import datetime
import functools

import torch
import torch.distributed as dist


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(args):
    def get_arg(key, default=None):
        if isinstance(args, dict):
            return args.get(key, default)
        return getattr(args, key, default)

    def set_arg(key, value):
        if isinstance(args, dict):
            args[key] = value
        else:
            setattr(args, key, value)

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        set_arg("rank", int(os.environ["RANK"]))
        set_arg("world_size", int(os.environ["WORLD_SIZE"]))
        set_arg("gpu", int(os.environ["LOCAL_RANK"]))
    elif "SLURM_PROCID" in os.environ:
        set_arg("rank", int(os.environ["SLURM_PROCID"]))
        set_arg("gpu", get_arg("rank") % torch.cuda.device_count())
    else:
        print("Not using distributed mode")
        set_arg("use_distributed", False)
        return

    set_arg("use_distributed", True)

    torch.cuda.set_device(get_arg("gpu"))
    set_arg("dist_backend", "nccl")
    
    dist_url = get_arg("dist_url", "env://")

    print(
        "| distributed init (rank {}, world {}): {}".format(
            get_arg("rank"), get_arg("world_size"), dist_url
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=get_arg("dist_backend"),
        init_method=dist_url,
        world_size=get_arg("world_size"),
        rank=get_arg("rank"),
        timeout=datetime.timedelta(
            days=365
        ),  # allow auto-downloading and de-compressing
    )
    torch.distributed.barrier()
    setup_for_distributed(get_arg("rank") == 0)


def get_dist_info():
    if torch.__version__ < "1.0":
        initialized = dist._initialized
    else:
        initialized = dist.is_initialized()
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:  # non-distributed training
        rank = 0
        world_size = 1
    return rank, world_size


def main_process(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper