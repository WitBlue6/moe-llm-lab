"""Plot DPO/reward/PPO/GRPO metrics without loading model checkpoints.

Example: uv run --locked --extra plot python scripts/plot_posttraining.py \
    --runs runs/post-ppo-001 --output reports/post-ppo-metrics.png
"""
import argparse
import json
import math
from pathlib import Path
import warnings


METRICS = ('train_reward', 'val_reward', 'reference_kl', 'val_kl',
           'truncated_fraction', 'clip_fraction', 'value_loss', 'mean_ratio',
           'zero_variance_group_fraction', 'train_preference_accuracy',
           'val_preference_accuracy', 'val_loss', 'train_loss', 'aux_loss')
TITLES = {
 'train_reward':'Training reward', 'val_reward':'Validation reward (evaluated steps only)',
 'reference_kl':'Training reference KL', 'val_kl':'Validation reference KL',
 'truncated_fraction':'Responses reaching generation limit', 'clip_fraction':'PPO clip fraction',
 'value_loss':'PPO value loss', 'mean_ratio':'Mean policy probability ratio',
 'zero_variance_group_fraction':'GRPO zero-variance groups',
 'train_preference_accuracy':'Training preference accuracy',
 'val_preference_accuracy':'Validation preference accuracy',
 'val_loss':'Validation preference loss', 'train_loss':'Training objective (not SFT CE)',
 'aux_loss':'MoE auxiliary loss'}


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
        raise ValueError(f'{path}: no complete post-training metric records')
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


def plot_runs(paths, output, labels=None, smooth_window=1, metrics=None):
    if smooth_window < 1:
        raise ValueError('smooth-window must be positive')
    if labels is not None and len(labels) != len(paths):
        raise ValueError('provide one label per run')
    output = Path(output)
    if output.suffix.lower() not in ('.png', '.pdf', '.svg'):
        raise ValueError('output must end in .png, .pdf or .svg')
    if output.exists():
        raise FileExistsError(f'use a new output filename: {output}')
    if metrics is not None and (not metrics or len(set(metrics)) != len(metrics) or any(k not in METRICS for k in metrics)):
        raise ValueError('metrics must be unique supported names')
    datasets = [read_metrics(path) for path in paths]
    selected = metrics or [k for k in METRICS if any(d[k][1] for d in datasets)]
    labels = labels or [str(Path(path).parent if Path(path).is_file() else Path(path)) for path in paths]
    # Import only when actually plotting; Agg also works on headless servers.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    nrows = math.ceil(len(selected)/2)
    fig, axes = plt.subplots(nrows, 2, figsize=(13, 3*nrows), squeeze=False, constrained_layout=True)
    flat = axes.flatten()
    for ax, key in zip(flat, selected):
        title = TITLES[key]
        found = False
        for index, (label, data) in enumerate(zip(labels, datasets)):
            steps, values = data[key]
            if not values:
                continue
            found = True
            color = f'C{index % 10}'
            if not key.startswith('val_') and smooth_window > 1:
                ax.plot(steps, values, color=color, alpha=.2, linewidth=.6)
                ax.plot(steps, moving_average(values, smooth_window), color=color,
                        linewidth=1.3, label=f'{label} (mean {smooth_window})')
            else:
                ax.plot(steps, values, color=color, linewidth=1.2,
                        marker='o' if key.startswith('val_') or len(values) == 1 else None,
                        markersize=3, label=label)
        ax.set_title(title, loc='left')
        ax.set_ylabel(key)
        ax.grid(alpha=.25)
        if found:
            ax.legend(fontsize=8)
        else:
            ax.text(.5, .5, 'Not recorded yet / not applicable', ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('Logged step (rollout iteration for PPO/GRPO)')
    for ax in flat[len(selected):]:
        ax.set_visible(False)
    fig.suptitle('Post-training metrics')
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Exclusive creation protects existing reports, including a concurrent writer.
        with output.open('xb') as handle:
            fig.savefig(handle, format=output.suffix[1:].lower(), dpi=160)
    finally:
        plt.close(fig)
    return {label: {key: len(data[key][0]) for key in selected} for label, data in zip(labels, datasets)}


def main():
    parser = argparse.ArgumentParser(description='Plot rewards, KL and diagnostics from post-training metrics.jsonl')
    parser.add_argument('--runs', nargs='+', required=True, help='run directories or metrics.jsonl files; one curve per input')
    parser.add_argument('--labels', nargs='+', help='optional legend labels, one per input')
    parser.add_argument('--output', required=True, help='new PNG, PDF or SVG output file')
    parser.add_argument('--smooth-window', type=int, default=1, help='trailing mean for non-validation metrics; raw traces remain visible')
    parser.add_argument('--metrics', nargs='+', choices=METRICS, help='optional panel selection; default shows recorded metrics only')
    args = parser.parse_args()
    counts = plot_runs(args.runs, args.output, args.labels, args.smooth_window, args.metrics)
    print(json.dumps({'output': str(Path(args.output).resolve()), 'points': counts}, indent=2))


if __name__ == '__main__':
    main()
