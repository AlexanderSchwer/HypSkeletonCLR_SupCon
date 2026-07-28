#!/usr/bin/env python
import os
import re
from datetime import datetime


WORK_DIR_MODES = ('auto', 'error', 'resume')


def prepare_training_work_dir(work_dir, mode='auto', run_id=None, run_subdir='runs'):
    """Return and reserve the effective work directory for a training run."""
    if mode not in WORK_DIR_MODES:
        raise ValueError('Unknown work_dir_mode: {}'.format(mode))

    work_dir = os.path.normpath(os.path.expandvars(os.path.expanduser(work_dir)))

    if mode == 'resume':
        return work_dir

    if mode == 'error':
        if os.path.exists(work_dir):
            raise FileExistsError('work_dir already exists: {}'.format(work_dir))
        os.makedirs(work_dir)
        return work_dir

    run_subdir = _clean_path_component(run_subdir, 'run_subdir')
    base_work_dir = _strip_existing_run_leaf(work_dir, run_subdir)
    run_id = _clean_path_component(run_id, 'run_id') if run_id else _default_run_id()
    return _reserve_unique_dir(os.path.join(base_work_dir, run_subdir, run_id))


def _default_run_id():
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    return '{}_pid{}'.format(timestamp, os.getpid())


def _reserve_unique_dir(first_candidate):
    for attempt in range(1000):
        candidate = first_candidate
        if attempt:
            candidate = '{}_{:03d}'.format(first_candidate, attempt)
        try:
            os.makedirs(candidate)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError('Could not allocate a unique work_dir below {}'.format(first_candidate))


def _strip_existing_run_leaf(work_dir, run_subdir):
    parent = os.path.dirname(work_dir)
    if os.path.basename(parent) == run_subdir:
        return os.path.dirname(parent)
    return work_dir


def _clean_path_component(value, name):
    value = str(value).strip()
    value = re.sub(r'[^A-Za-z0-9_.-]+', '-', value).strip('.-')
    if not value:
        raise ValueError('{} must not be empty'.format(name))
    if value in ('.', '..') or os.path.basename(value) != value:
        raise ValueError('{} must be a single path component'.format(name))
    return value
