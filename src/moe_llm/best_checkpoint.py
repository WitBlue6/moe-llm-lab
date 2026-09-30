"""Track the lowest finite validation loss, including across exact resumes."""
import json
import math
from pathlib import Path

import torch.distributed as dist


class BestCheckpoint:
    def __init__(self, state, scope):
        self.best = dict(state['best']) if state and state.get('best') else None
        self.scope = scope
        if self.best:
            if self.best['validation_scope'] != scope:
                raise ValueError('best-checkpoint resume requires unchanged validation scope')
            if not Path(self.best['checkpoint']).is_file():
                raise FileNotFoundError(f"historical best checkpoint is missing: {self.best['checkpoint']}")

    def consider(self, metrics, root, rank=0, world=1):
        improved = False
        if rank == 0:
            loss = metrics.get('val_loss')
            if loss is not None and math.isfinite(loss) and (self.best is None or loss < self.best['val_loss']):
                self.best = {'metric': 'val_loss', 'mode': 'min', 'val_loss': float(loss),
                             'step': metrics['step'],
                             'checkpoint': str((Path(root) / f"step-{metrics['step']:07d}.pt").resolve()),
                             'validation_scope': self.scope}
                improved = True
        if world > 1:
            # SigLIP evaluates on rank zero only. All ranks must agree to enter
            # checkpoint RNG gathering, including on non-periodic best steps.
            message = [improved, self.best]
            dist.broadcast_object_list(message, src=0)
            improved, self.best = message
        return improved

    def publish(self, root):
        if self.best is None:
            return
        if not Path(self.best['checkpoint']).is_file():
            raise FileNotFoundError(self.best['checkpoint'])
        path = Path(root) / 'best.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.best, indent=2) + '\n')
        temporary.replace(path)
