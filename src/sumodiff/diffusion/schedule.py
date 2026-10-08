"""An update-indexed, serializable linear-warmup then cosine schedule."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math


@dataclass(frozen=True)
class ScheduleConfig:
    schema_version: str = 'sumodiff.warmup.cosine.config.v1'
    peak_lr: float = 1e-4
    minimum_lr: float = 1e-6
    warmup_updates: int = 600
    total_updates: int = 15000
    auxiliary_warmup_updates: int = 1500

    def __post_init__(self):
        if self.schema_version != 'sumodiff.warmup.cosine.config.v1':
            raise ValueError('Unsupported schedule schema')
        if any(type(x) is not int or x<1 for x in (self.warmup_updates,self.total_updates,self.auxiliary_warmup_updates)):
            raise ValueError('Schedule budgets must be positive integer updates')
        if not 2 <= self.warmup_updates < self.total_updates or self.auxiliary_warmup_updates > self.total_updates:
            raise ValueError('Invalid warmup/total budgets')
        if any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in (self.peak_lr,self.minimum_lr)):
            raise ValueError('Invalid schedule learning rates')
        if self.minimum_lr >= self.peak_lr:
            raise ValueError('Minimum lr must be below peak lr')

    def to_dict(self):
        return asdict(self)


def resolve_schedule(value):
    if not isinstance(value,dict) or set(value)-set(ScheduleConfig.__dataclass_fields__):
        raise ValueError('Unknown learning-rate schedule options')
    return ScheduleConfig(**value)


class UpdateSchedule:
    """Update 1 uses min_lr, warmup-end peak_lr, final update min_lr.

    The auxiliary coefficient is min(update/auxiliary_warmup,1). The budget
    stays at 15000 even in a short prefix; recovery never restarts either ramp.
    """
    def __init__(self,config):
        self.config = config
        self.completed_updates = 0
        self.last_lr = None
        self.last_auxiliary_ramp = None

    def learning_rate(self,update):
        c = self.config
        if type(update) is not int or not 1 <= update <= c.total_updates:
            raise ValueError('Update outside schedule budget')
        if update <= c.warmup_updates:
            return c.minimum_lr+(c.peak_lr-c.minimum_lr)*(update-1)/(c.warmup_updates-1)
        progress = (update-c.warmup_updates)/(c.total_updates-c.warmup_updates)
        return c.minimum_lr+(c.peak_lr-c.minimum_lr)*.5*(1+math.cos(math.pi*progress))

    def auxiliary_ramp(self,update):
        self.learning_rate(update)  # Validate the same update index.
        return min(update/self.config.auxiliary_warmup_updates,1.)

    def apply(self,optimizer,update):
        if update != self.completed_updates+1:
            raise ValueError('Scheduler updates must be consecutive')
        lr = self.learning_rate(update)
        for group in optimizer.param_groups:
            group['lr'] = lr
        return lr,self.auxiliary_ramp(update)

    def complete(self,update):
        if update != self.completed_updates+1:
            raise ValueError('Invalid completed scheduler update')
        self.completed_updates = update
        self.last_lr = self.learning_rate(update)
        self.last_auxiliary_ramp = self.auxiliary_ramp(update)

    def state_dict(self):
        signature = hashlib.sha256(json.dumps(self.config.to_dict(),sort_keys=True).encode()).hexdigest()
        return dict(schema_version='sumodiff.warmup.cosine.state.v1',config_sha256=signature,
                    completed_updates=self.completed_updates,last_lr=self.last_lr,
                    last_auxiliary_ramp=self.last_auxiliary_ramp)

    def load_state_dict(self,value,optimizer=None,expected_updates=None):
        if not isinstance(value,dict) or set(value) != set(self.state_dict()):
            raise ValueError('Incomplete scheduler state')
        n = value['completed_updates']
        if type(n) is not int or not 0 <= n <= self.config.total_updates:
            raise ValueError('Invalid scheduler state budget')
        if expected_updates is not None and n != expected_updates:
            raise ValueError('Checkpoint/scheduler update mismatch')
        clone = UpdateSchedule(self.config)
        clone.completed_updates = n
        if n:
            clone.last_lr = clone.learning_rate(n)
            clone.last_auxiliary_ramp = clone.auxiliary_ramp(n)
        if value != clone.state_dict():
            raise ValueError('Scheduler state/config/ramp mismatch')
        if optimizer is not None and n and any(g['lr']!=value['last_lr'] for g in optimizer.param_groups):
            raise ValueError('Optimizer learning rate differs from saved scheduler')
        self.completed_updates = n
        self.last_lr = value['last_lr']
        self.last_auxiliary_ramp = value['last_auxiliary_ramp']
