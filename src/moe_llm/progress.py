"""Rank-zero terminal progress; JSON logs stay usable in redirected output."""
import json
import os
import shutil
import sys
import time


_active = None


def log_json(record):
    """Print an event above the live line, without modifying the logged record."""
    if _active is not None:
        _active.log(record)
    else:
        print(json.dumps(record), flush=True)


class TrainingProgress:
    def __init__(self, *, rank, total, start, steps_per_epoch, epochs=None, stream=None):
        self.stream = sys.stdout if stream is None else stream
        self.rank, self.total, self.step = rank, total, start
        self.steps_per_epoch, self.epochs = steps_per_epoch, epochs
        mode = os.environ.get('MOE_LAB_PROGRESS', 'auto')
        self.enabled = rank == 0 and mode != '0' and (mode == '1' or self.stream.isatty())
        self.start, self.started = start, time.monotonic()
        self.train_loss = self.val_loss = self.val_step = None
        self.status = 'train'

    def __enter__(self):
        global _active
        self.previous = _active
        if self.rank == 0:
            _active = self
        self.render()
        return self

    def __exit__(self, exc_type, exc, tb):
        global _active
        self.status = 'failed' if exc_type else ('done' if self.step >= self.total else 'stopped')
        self.render()
        if self.enabled:
            self.stream.write('\n')
            self.stream.flush()
        if self.rank == 0:
            _active = self.previous

    def phase(self, status):
        self.status = status
        self.render()

    def update(self, metrics):
        self.step = metrics['step']
        self.train_loss = metrics.get('train_loss', self.train_loss)
        if 'val_loss' in metrics:
            self.val_loss, self.val_step = metrics['val_loss'], self.step
        self.render()

    def description(self):
        if self.epochs is not None:
            # At a completed boundary keep the just-finished epoch at 100%.
            epoch = min(self.epochs, max(0, self.step - 1) // self.steps_per_epoch + 1)
            current = self.step - (epoch - 1) * self.steps_per_epoch
            label, total = f'Epoch {epoch}/{self.epochs}', self.steps_per_epoch
        else:
            label, current, total = 'Steps', self.step, self.total
        filled = min(12, int(12 * current / max(total, 1)))
        bar = '#' * filled + '-' * (12 - filled)
        train = '--' if self.train_loss is None else f'{self.train_loss:.4f}'
        val = '--' if self.val_loss is None else f'{self.val_loss:.4f}@{self.val_step}'
        rate = (self.step - self.start) / max(time.monotonic() - self.started, 1e-9)
        eta = '--' if rate <= 0 else f'{int(max(0, total-current) / rate)}s'
        return f'{label} [{bar}] {current}/{total} train={train} val={val} {self.status} ETA={eta}'

    def clear(self):
        if self.enabled:
            self.stream.write('\r\033[2K')

    def render(self):
        if self.enabled:
            width = max(20, shutil.get_terminal_size(fallback=(120, 24)).columns - 1)
            self.clear()
            self.stream.write(self.description()[:width])
            self.stream.flush()

    def log(self, record):
        if self.rank != 0:
            return
        if 'train_loss' in record and 'step' in record:
            self.update(record)
        if record.get('event') == 'validation_start':
            self.status = 'validate'
        self.clear()
        self.stream.write(json.dumps(record) + '\n')
        self.stream.flush()
        self.render()
