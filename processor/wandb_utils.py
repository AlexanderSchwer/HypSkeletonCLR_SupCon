import os

import wandb

from .work_dir import wandb_run_identity_from_work_dir


DEFAULT_WANDB_PROJECT = 'HypSkeletonCLR_SupCon'


def init_wandb_from_work_dir(arg, job_type=None, config=None, **kwargs):
    if bool(getattr(arg, 'wandb_disabled', False)):
        return None

    name, group = wandb_run_identity_from_work_dir(
        getattr(arg, 'work_dir', ''),
        getattr(arg, 'run_subdir', 'runs'),
    )

    init_kwargs = {
        'project': os.environ.get('WANDB_PROJECT', DEFAULT_WANDB_PROJECT),
        'name': name,
        'config': vars(arg) if config is None else config,
    }
    if group:
        init_kwargs['group'] = group
    if job_type:
        init_kwargs['job_type'] = job_type
    if bool(getattr(arg, 'wandb_offline', False)):
        init_kwargs['mode'] = 'offline'

    init_kwargs.update(kwargs)
    return wandb.init(**init_kwargs)
