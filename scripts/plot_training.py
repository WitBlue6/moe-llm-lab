"""Plot existing JSONL training metrics without loading model checkpoints.

Example: uv run --locked --extra plot python scripts/plot_training.py \
    --runs runs/text-pretrain-001 --output reports/pretrain-loss.png
"""
import argparse
import json
import math
from pathlib import Path
import warnings


METRICS = ('train_loss', 'aux_loss', 'val_loss')


def read_metrics(path):
    """Read a snapshot; ignore only an incomplete final JSONL line."""
    path = Path(path)
    if path.is_dir():
        path = path / 'metrics.jsonl'
    series = {key: ([], []) for key in METRICS}
    previous = -1
    with path.open(encoding='utf-8') as handle:
        for number, line in enumerate(handle, 1):
            # Writers append a newline for each complete record. During a live
            # run a partial final record may be visible; never plot that record.
            if not line.endswith('\n'):
                warnings.warn(f'{path}:{number}: ignored final line without newline')
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                step = row['step']
                if type(step) is not int or step < 0 or step <= previous:
                    raise ValueError('steps must be nonnegative, strictly increasing integers')
                previous = step
                for key in METRICS:
                    if key not in row:
                        continue
                    value = row[key]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError(f'{key} must be a finite number')
                    series[key][0].append(step)
                    series[key][1].append(value)
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f'{path}:{number}: invalid metrics: {error}') from error
    if not any(values for _, values in series.values()):
        raise ValueError(f'{path}: no complete loss records')
    return series


def moving_average(values, window):
    """Trailing mean over recorded updates, with shorter initial windows."""
    total = 0.
    result = []
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        result.append(total / min(index + 1, window))
    return result


def plot_runs(paths, output, labels=None, smooth_window=1):
    if smooth_window < 1:
        raise ValueError('smooth-window must be positive')
    if labels is not None and len(labels) != len(paths):
        raise ValueError('provide one label per run')
    output = Path(output)
    if output.suffix.lower() not in ('.png', '.pdf', '.svg'):
        raise ValueError('output must end in .png, .pdf or .svg')
    if output.exists():
        raise FileExistsError(f'use a new output filename: {output}')
    datasets = [read_metrics(path) for path in paths]
    labels = labels or [str(Path(path).parent if Path(path).is_file() else Path(path)) for path in paths]
    # Import only when actually plotting; Agg also works on headless servers.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    titles = ('Training loss', 'MoE auxiliary loss', 'Validation loss (evaluated steps only)')
    for ax, key, title in zip(axes, METRICS, titles):
        found = False
        for index, (label, data) in enumerate(zip(labels, datasets)):
            steps, values = data[key]
            if not values:
                continue
            found = True
            color = f'C{index % 10}'
            if key != 'val_loss' and smooth_window > 1:
                ax.plot(steps, values, color=color, alpha=.2, linewidth=.6)
                ax.plot(steps, moving_average(values, smooth_window), color=color,
                        linewidth=1.3, label=f'{label} (mean {smooth_window})')
            else:
                ax.plot(steps, values, color=color, linewidth=1.2,
                        marker='o' if key == 'val_loss' or len(values) == 1 else None,
                        markersize=3, label=label)
        ax.set_title(title, loc='left')
        ax.set_ylabel(key)
        ax.grid(alpha=.25)
        if found:
            ax.legend(fontsize=8)
        else:
            ax.text(.5, .5, 'Not recorded yet / not applicable', ha='center', va='center', transform=ax.transAxes)
    axes[-1].set_xlabel('Global optimizer step (within each run)')
    fig.suptitle('Training metrics')
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Exclusive creation protects existing reports, including a concurrent writer.
        with output.open('xb') as handle:
            fig.savefig(handle, format=output.suffix[1:].lower(), dpi=160)
    finally:
        plt.close(fig)
    return {label: {key: len(data[key][0]) for key in METRICS} for label, data in zip(labels, datasets)}


def main():
    parser = argparse.ArgumentParser(description='Plot train/aux/validation losses from metrics.jsonl')
    parser.add_argument('--runs', nargs='+', required=True, help='run directories or metrics.jsonl files; one curve per input')
    parser.add_argument('--labels', nargs='+', help='optional legend labels, one per input')
    parser.add_argument('--output', required=True, help='new PNG, PDF or SVG output file')
    parser.add_argument('--smooth-window', type=int, default=1, help='trailing mean for train/aux only; raw traces remain visible')
    args = parser.parse_args()
    counts = plot_runs(args.runs, args.output, args.labels, args.smooth_window)
    print(json.dumps({'output': str(Path(args.output).resolve()), 'points': counts}, indent=2))


if __name__ == '__main__':
    main()
