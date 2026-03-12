import argparse
import hashlib
import math
import os

import torch
import torch.distributed as dist
import swanlab
from PIL import Image
from torchvision.utils import make_grid


def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }


def generate_run_id(exp_name: str) -> str:
    # SwanLab resume requires a 21-character id. Make a deterministic 21-char id from exp_name.
    return hashlib.sha256(exp_name.encode('utf-8')).hexdigest()[:21]


def initialize(args, entity, exp_name, project_name):
    config_dict = namespace_to_dict(args)

    api_key = os.environ.get('SWANLAB_API_KEY', None)
    host = os.environ.get('SWANLAB_HOST', None)
    if api_key:
        if host:
            swanlab.login(api_key=api_key, host=host)
        else:
            swanlab.login(api_key=api_key)

    init_kwargs = dict(
        project=project_name,
        experiment_name=exp_name,
        config=config_dict,
        id=generate_run_id(exp_name),
        resume='allow',
    )
    if entity:
        init_kwargs['workspace'] = entity
    if getattr(args, 'local_out_path', None):
        init_kwargs['logdir'] = os.path.join(args.local_out_path, 'swanlog')
    mode = os.environ.get('SWANLAB_MODE', None)
    if mode:
        init_kwargs['mode'] = mode

    swanlab.init(**init_kwargs)


def log(stats, step=None):
    if is_main_process():
        swanlab.log({k: v for k, v in stats.items()}, step=step)


def log_image(name, sample, step=None):
    if is_main_process():
        sample = array2grid(sample)
        swanlab.log({f'{name}': swanlab.Image(sample), 'train_step': step}, step=step)


def finish():
    if is_main_process():
        swanlab.finish()


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x, nrow=nrow, normalize=True, value_range=(-1, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x
