from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    LambdaLR,
    LinearLR,
    MultiStepLR,
    SequentialLR,
    StepLR,
)


LR_SCHEDULERS = (
    'auto',
    'none',
    'multistep',
    'step',
    'exponential',
    'cosine',
    'cosine_warm_restarts',
)


def add_lr_scheduler_args(parser):
    parser.add_argument('--lr_scheduler', default='auto', choices=LR_SCHEDULERS,
                        help='learning-rate scheduler: auto maps step to multistep')
    parser.add_argument('--lr_milestones', type=int, default=[], nargs='+',
                        help='epoch milestones for MultiStepLR; defaults to step when empty')
    parser.add_argument('--lr_step_size', type=int, default=250,
                        help='step size in epochs for StepLR')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='multiplicative decay factor for StepLR, MultiStepLR, and ExponentialLR')
    parser.add_argument('--lr_warmup_epochs', type=int, default=0,
                        help='linear warmup epochs before the selected scheduler')
    parser.add_argument('--lr_warmup_start_factor', type=float, default=0.001,
                        help='initial LR factor for LinearLR warmup')
    parser.add_argument('--lr_eta_min', type=float, default=None,
                        help='minimum learning rate for cosine schedulers; null defaults to 0.01 * base_lr')
    parser.add_argument('--lr_t_max', type=int, default=None,
                        help='T_max for CosineAnnealingLR; null defaults to min(50, num_epoch - lr_warmup_epochs)')
    parser.add_argument('--lr_t_0', type=int, default=10,
                        help='initial cycle length for CosineAnnealingWarmRestarts')
    parser.add_argument('--lr_t_mult', type=int, default=1,
                        help='cycle length multiplier for CosineAnnealingWarmRestarts')


class LRSchedulerMixin:
    def load_lr_scheduler(self):
        self.lr_scheduler = self._build_lr_scheduler()
        self.lr = self._current_lr()

    def adjust_lr_scheduler(self):
        self.adjust_lr()

    def adjust_lr(self):
        self.lr = self._current_lr()

    def after_train_epoch(self, epoch):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def _current_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def _build_lr_scheduler(self):
        scheduler_name = self._lr_scheduler_name()
        warmup_epochs = max(0, int(getattr(self.arg, 'lr_warmup_epochs', 0)))
        main_scheduler = self._build_main_lr_scheduler(scheduler_name, warmup_epochs)

        if warmup_epochs == 0:
            return main_scheduler

        warmup = LinearLR(
            self.optimizer,
            start_factor=float(self.arg.lr_warmup_start_factor),
            total_iters=warmup_epochs,
        )
        if main_scheduler is None:
            return warmup
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup, main_scheduler],
            milestones=[warmup_epochs],
        )

    def _build_main_lr_scheduler(self, scheduler_name, warmup_epochs):
        if scheduler_name == 'none':
            return None

        if scheduler_name == 'multistep':
            milestones = self._lr_milestones(warmup_epochs)
            if not milestones:
                return None
            return MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=float(self.arg.lr_gamma),
            )

        if scheduler_name == 'step':
            step_size = max(1, int(self.arg.lr_step_size))
            return StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=float(self.arg.lr_gamma),
            )

        if scheduler_name == 'exponential':
            return ExponentialLR(
                self.optimizer,
                gamma=float(self.arg.lr_gamma),
            )

        if scheduler_name == 'cosine':
            eta_min = self._lr_eta_min()
            t_max = self._lr_t_max(warmup_epochs)
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=eta_min,
            )
            schedule_epochs = self._lr_schedule_epochs(warmup_epochs)
            if t_max >= schedule_epochs:
                return scheduler
            eta_min_factor = eta_min / float(self.arg.base_lr)
            return SequentialLR(
                self.optimizer,
                schedulers=[
                    scheduler,
                    LambdaLR(self.optimizer, lr_lambda=lambda _: eta_min_factor),
                ],
                milestones=[t_max],
            )

        if scheduler_name == 'cosine_warm_restarts':
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=max(1, int(self.arg.lr_t_0)),
                T_mult=max(1, int(self.arg.lr_t_mult)),
                eta_min=self._lr_eta_min(),
            )

        raise ValueError('Unknown lr_scheduler: {}'.format(scheduler_name))

    def _lr_scheduler_name(self):
        scheduler_name = str(getattr(self.arg, 'lr_scheduler', 'auto')).lower()
        scheduler_name = scheduler_name.replace('-', '_')

        if scheduler_name == 'auto':
            if getattr(self.arg, 'step', None):
                return 'multistep'
            return 'none'

        if scheduler_name not in LR_SCHEDULERS:
            raise ValueError('Unknown lr_scheduler: {}'.format(scheduler_name))
        return scheduler_name

    def _lr_milestones(self, warmup_epochs):
        milestones = list(getattr(self.arg, 'lr_milestones', None) or [])
        if not milestones:
            milestones = list(getattr(self.arg, 'step', None) or [])
        if warmup_epochs == 0:
            return [int(value) for value in milestones]
        return [max(1, int(value) - warmup_epochs) for value in milestones if int(value) > warmup_epochs]

    def _lr_eta_min(self):
        eta_min = getattr(self.arg, 'lr_eta_min', None)
        if eta_min is None:
            return float(self.arg.base_lr) * 0.01
        return float(eta_min)

    def _lr_t_max(self, warmup_epochs):
        t_max = getattr(self.arg, 'lr_t_max', None)
        if t_max is not None:
            return max(1, int(t_max))
        return min(50, self._lr_schedule_epochs(warmup_epochs))

    def _lr_schedule_epochs(self, warmup_epochs):
        return max(1, int(self.arg.num_epoch) - warmup_epochs)
