#!/usr/bin/env python
import os
import re
from datetime import datetime


WORK_DIR_MODES = ('auto', 'error', 'resume')
WORK_DIR_LAYOUTS = ('canonical', 'manual')

_RUN_FAMILIES = (
    'crossclr_3views',
    'crossclr',
    'linear_eval',
    'plotting',
    'skeletonclr_3views_eucl',
    'skeletonclr_eucl_3views',
    'skeletonclr_3views',
    'skeletonclr_hyptorch',
    'skeletonclr_gradfix',
    'skeletonclr_eucl',
    'skeletonclr_att',
    'skeletonclr',
)

_SKELETONCLR_FAMILY_ALIASES = (
    ('skeletonclr_3views_eucl', 'skeletonclr_3views_eucl'),
    ('skeletonclr_eucl_3views', 'skeletonclr_3views_eucl'),
    ('skeletonclr_3views', 'skeletonclr_3views'),
    ('skeletonclr_gradfix', 'skeletonclr_gradfix'),
    ('skeletonclr_eucl', 'skeletonclr_eucl'),
    ('skeletonclr_att', 'skeletonclr_att'),
)

_PSEUDO_CONTRASTIVE_MODES = ('pseudo_hard', 'pseudo_soft')


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


def build_canonical_work_dir(arg, processor_name=None):
    """Build a stable experiment base path from the actual run arguments."""
    family = _infer_run_family(arg, processor_name)
    root = _infer_work_dir_root(getattr(arg, 'work_dir', 'work_dir'), family)
    setup = _build_setup_component(arg)
    experiment = _build_experiment_component(arg, family)
    return os.path.join(root, family, setup, experiment)


def wandb_run_identity_from_work_dir(work_dir, run_subdir='runs'):
    """Return a W&B name from a finalized work_dir path, without grouping."""
    parts = _path_parts(os.path.normpath(os.path.expandvars(os.path.expanduser(work_dir))))
    if 'work_dir' in parts:
        parts = parts[parts.index('work_dir') + 1:]

    if not parts:
        name = os.path.basename(str(work_dir).rstrip('/\\')) or 'run'
        return name, None

    name_parts = parts
    if len(parts) >= 2 and parts[-2] == run_subdir:
        name_parts = parts[:-2]
    name = '/'.join(name_parts) if name_parts else '/'.join(parts)
    return name, None


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


def _infer_run_family(arg, processor_name=None):
    haystack = ' '.join([
        str(getattr(arg, 'config', '') or ''),
        str(getattr(arg, 'model', '') or ''),
        str(getattr(arg, 'work_dir', '') or ''),
        str(processor_name or ''),
    ]).lower()

    if 'crossclr_3views' in haystack:
        return 'crossclr_3views'
    if 'crossclr' in haystack:
        return 'crossclr'
    if 'linear_eval' in haystack or processor_name == 'LE_Processor':
        return 'linear_eval'
    if 'plot' in haystack:
        return 'plotting'
    for needle, family in _SKELETONCLR_FAMILY_ALIASES:
        if needle in haystack:
            return family
    if 'skeletonclr' in haystack:
        return 'skeletonclr'
    return 'training'


def _infer_work_dir_root(work_dir, family):
    work_dir = os.path.normpath(os.path.expandvars(os.path.expanduser(work_dir)))
    drive, tail = os.path.splitdrive(work_dir)
    parts = [part for part in re.split(r'[\\/]+', tail) if part]

    if 'work_dir' in parts:
        index = parts.index('work_dir')
        return _join_path_prefix(drive, os.path.isabs(work_dir), parts[:index + 1])

    for run_family in _RUN_FAMILIES + (family,):
        if run_family in parts:
            index = parts.index(run_family)
            prefix = parts[:index] or ['work_dir']
            return _join_path_prefix(drive, os.path.isabs(work_dir), prefix)

    if os.path.basename(work_dir) == family:
        return os.path.dirname(work_dir) or '.'
    return work_dir


def _join_path_prefix(drive, is_absolute, parts):
    if not parts:
        return drive + os.sep if is_absolute else (drive or '.')
    prefix = os.path.join(*parts)
    if is_absolute:
        root = drive + os.sep if drive else os.sep
        return os.path.join(root, prefix)
    if drive:
        return drive + prefix
    return prefix


def _path_parts(path):
    drive, tail = os.path.splitdrive(path)
    return [part for part in re.split(r'[\\/]+', tail) if part]


def _build_setup_component(arg):
    model_args = getattr(arg, 'model_args', {}) or {}
    data_paths = _feeder_arg_strings(arg)
    config_paths = _config_arg_strings(arg)
    work_dir_paths = [str(getattr(arg, 'work_dir', '') or '')]
    paths = data_paths + config_paths + work_dir_paths

    dataset = _infer_dataset(model_args, paths)
    split = (_infer_split(data_paths) or
             _infer_split(config_paths) or
             _infer_split(work_dir_paths))
    frame_count = (_infer_frame_count(data_paths) or
                   _infer_frame_count(config_paths) or
                   _infer_frame_count(work_dir_paths))
    hidden_channels = model_args.get('hidden_channels')

    parts = [dataset]
    if split:
        parts.append(split)
    if frame_count:
        parts.append('frame{}'.format(frame_count))
    if hidden_channels is not None:
        parts.append('hc{}'.format(hidden_channels))
    return _slug('-'.join(parts))


def _build_experiment_component(arg, family=None):
    model_args = getattr(arg, 'model_args', {}) or {}
    tags = []

    geometry_impl = _infer_geometry_impl(arg)
    if geometry_impl:
        tags.append(geometry_impl)
        curvature = model_args.get('curvature', getattr(arg, 'curvature', None))
        if curvature is not None:
            tags.append('c{}'.format(_format_value(curvature)))

    if bool(model_args.get('cluster_enabled', False)):
        num_clusters = model_args.get('num_clusters')
        if num_clusters is not None:
            tags.append('clust{}'.format(num_clusters))

    if _uses_pseudo_labels(arg):
        tags.append('pseudo')

    base_lr = getattr(arg, 'base_lr', None)
    if base_lr is not None:
        tags.append('lrb{}'.format(_format_value(base_lr)))

    if family == 'linear_eval':
        num_epoch = getattr(arg, 'num_epoch', None)
        if num_epoch is not None:
            tags.append('ep{}'.format(_format_value(num_epoch)))

    scheduler_tag = _infer_scheduler_tag(arg)
    if scheduler_tag:
        tags.append(scheduler_tag)
        if family == 'linear_eval':
            tags.extend(_linear_eval_scheduler_detail_tags(arg, scheduler_tag))

    if not tags:
        tags.append('default')
    return _slug('-'.join(tags))


def _uses_pseudo_labels(arg):
    mode = getattr(arg, 'contrastive_mode', None)
    if mode in _PSEUDO_CONTRASTIVE_MODES:
        return True
    return any(
        mode in _PSEUDO_CONTRASTIVE_MODES
        for mode in _contrastive_schedule_modes(getattr(arg, 'contrastive_schedule', None))
    )


def _contrastive_schedule_modes(schedule):
    if schedule is None:
        return []
    if isinstance(schedule, str):
        return [
            mode for mode in _PSEUDO_CONTRASTIVE_MODES
            if mode in schedule.lower()
        ]
    if isinstance(schedule, dict):
        schedule = [schedule]
    if not isinstance(schedule, list):
        return []
    return [
        phase.get('mode') for phase in schedule
        if isinstance(phase, dict)
    ]


def _linear_eval_scheduler_detail_tags(arg, scheduler_tag):
    tags = []
    if scheduler_tag == 'multistep':
        milestones = list(getattr(arg, 'lr_milestones', None) or [])
        if not milestones:
            milestones = list(getattr(arg, 'step', None) or [])
        if milestones:
            tags.append('ms{}'.format('-'.join(str(int(value)) for value in milestones)))
        gamma = getattr(arg, 'lr_gamma', None)
        if gamma is not None:
            tags.append('g{}'.format(_format_value(gamma)))
    elif scheduler_tag == 'step':
        step_size = getattr(arg, 'lr_step_size', None)
        if step_size is not None:
            tags.append('step{}'.format(_format_value(step_size)))
        gamma = getattr(arg, 'lr_gamma', None)
        if gamma is not None:
            tags.append('g{}'.format(_format_value(gamma)))
    elif scheduler_tag == 'exponential':
        gamma = getattr(arg, 'lr_gamma', None)
        if gamma is not None:
            tags.append('g{}'.format(_format_value(gamma)))
    elif scheduler_tag == 'cosine':
        t_max = getattr(arg, 'lr_t_max', None)
        if t_max is not None:
            tags.append('tmax{}'.format(_format_value(t_max)))
        eta_min = getattr(arg, 'lr_eta_min', None)
        if eta_min is not None:
            tags.append('eta{}'.format(_format_value(eta_min)))
    elif scheduler_tag == 'cosine-warm-restarts':
        t_0 = getattr(arg, 'lr_t_0', None)
        if t_0 is not None:
            tags.append('t0{}'.format(_format_value(t_0)))
        t_mult = getattr(arg, 'lr_t_mult', None)
        if t_mult is not None:
            tags.append('tmult{}'.format(_format_value(t_mult)))
        eta_min = getattr(arg, 'lr_eta_min', None)
        if eta_min is not None:
            tags.append('eta{}'.format(_format_value(eta_min)))
    return tags


def _infer_dataset(model_args, paths):
    joined = ' '.join(paths).lower()
    match = re.search(r'ntu[-_]?rgbd?[-_]?(\d+)|ntu(\d+)', joined)
    if match:
        return 'ntu{}'.format(match.group(1) or match.group(2))

    num_class = model_args.get('num_class')
    if num_class in (60, 120):
        return 'ntu{}'.format(num_class)
    return 'dataset'


def _infer_split(paths):
    joined = ' '.join(paths).lower()
    if re.search(r'(^|[\\/_.-])xview($|[\\/_.-])', joined):
        return 'xview'
    if re.search(r'(^|[\\/_.-])xsub($|[\\/_.-])', joined):
        return 'xsubject'
    if 'xsubject' in joined:
        return 'xsubject'
    return None


def _infer_frame_count(paths):
    for value in paths:
        match = re.search(r'frame[_-]?(\d+)|frame(\d+)', value.lower())
        if match:
            return match.group(1) or match.group(2)
    for value in paths:
        normalized = value.lower().replace('\\', '/')
        if re.search(r'(^|/)(train|val|test)_(position|motion)\.npy$', normalized):
            return '50'
    return None


def _infer_geometry_impl(arg):
    haystack = ' '.join([
        str(getattr(arg, 'config', '') or ''),
        str(getattr(arg, 'model', '') or ''),
        str(getattr(arg, 'processor', '') or ''),
    ]).lower()

    if 'eucl' in haystack:
        return None
    if 'hyptorch' in haystack:
        return 'hyptorch'
    if 'skeletonclr' in haystack or 'crossclr' in haystack:
        return 'geoopt'
    return None


def _infer_scheduler_tag(arg):
    scheduler = str(getattr(arg, 'lr_scheduler', 'auto') or 'auto').lower()
    scheduler = scheduler.replace('-', '_')
    if scheduler == 'auto':
        if getattr(arg, 'lr_milestones', None) or getattr(arg, 'step', None):
            scheduler = 'multistep'
        else:
            scheduler = 'none'
    if scheduler == 'none':
        return 'constant-lr'
    return scheduler.replace('_', '-')


def _config_arg_strings(arg):
    values = [
        getattr(arg, 'config', ''),
        getattr(arg, 'model', ''),
    ]
    return [str(value) for value in values if value is not None]


def _feeder_arg_strings(arg):
    values = []
    for attr in ('train_feeder_args', 'test_feeder_args'):
        feeder_args = getattr(arg, attr, None) or {}
        if isinstance(feeder_args, dict):
            values.extend(str(value) for value in feeder_args.values())
    return [str(value) for value in values if value is not None]


def _format_value(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if number.is_integer():
        return str(int(number))
    return ('{:.8g}'.format(number)).replace('.', 'p').replace('-', 'm')


def _slug(value):
    value = re.sub(r'[^A-Za-z0-9_.-]+', '-', value).strip('.-')
    value = re.sub(r'-+', '-', value)
    return value.lower()
