#!/usr/bin/env python3
"""Plot already-recorded two-head A losses; never connect to or change training."""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = {'dinov3_l': '#2563a6', 'qwen3vl_4b': '#d16b28',
          'siglip2': '#318564', 'aimv2_huge': '#8558a0'}
LABELS = {'dinov3_l': 'DINOv3', 'qwen3vl_4b': 'Qwen-VL',
          'siglip2': 'SigLIP2', 'aimv2_huge': 'AIMv2'}
INITIAL_SOURCE = 'a25e4c42794b13d5c881d3f1994944929b019190'
RESUMED_SOURCE = '1e06bf49569b4ee7847a417bc1ff1445bb8ba51b'


def smooth(values, width=5):
    return np.array([np.mean(values[max(0, i-width+1):i+1]) for i in range(len(values))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metrics', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smooth', type=int, default=10)
    args = parser.parse_args()
    assert args.smooth >= 1
    data = json.loads(args.metrics.read_text())
    rows = data['rows']
    steps = np.array([row['step'] for row in rows])
    assert steps.tolist() == list(range(1, len(rows)+1))
    resumed = data['source_commit'] == RESUMED_SOURCE
    if resumed:
        assert len(rows) > 200
        assert data['trajectory_sources'] == [
            {'steps': [1, 200], 'commit': INITIAL_SOURCE},
            {'steps': [201, len(rows)], 'commit': RESUMED_SOURCE}]
        assert data['resume200']['step'] == 200
        assert data['resume200']['G_Adam_reset'] is True
    else:
        assert data['source_commit'] == INITIAL_SOURCE and len(rows) <= 200
    has_adv = any(row.get('s2/gan/g_active') for row in rows)
    events = ['Dashed: GAN-G starts at 51']
    events.extend(f"Dotted: D LR {a['old_lr']:g} to {a['new_lr']:g} from {a['effective_from_step']}"
                  for a in data.get('D_lr_actions', []))
    event_lines = textwrap.wrap('     |     '.join(events), width=150)
    if resumed:
        event_lines.append('Dash-dot: resume after 200; G Adam history reset. Moving means restart at 201.')
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11,
        'axes.labelsize': 10, 'legend.fontsize': 9, 'axes.spines.top': False,
        'axes.spines.right': False, 'figure.facecolor': 'white',
        'savefig.facecolor': 'white', 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(12.5, 14 if has_adv else 11))
    grid = fig.add_gridspec(4 if has_adv else 3, 2,
        height_ratios=[1, 1, 1, .9] if has_adv else [1, 1, .8],
        left=.09, right=.975, top=.89-.017*(len(event_lines)-1), bottom=.10,
        hspace=.42, wspace=.25)
    axes = []

    def series(ax, key, color, label, active=False):
        valid = [row for row in rows if key in row and
                 (not active or row.get('s2/gan/g_active'))]
        if not valid:
            return
        x = np.array([row['step'] for row in valid])
        y = np.array([row[key] for row in valid], dtype=float)
        assert np.isfinite(y).all(), key
        # Keep the optimizer reset visible rather than smoothing across it.
        groups = [x <= 200, x > 200] if resumed else [np.ones(len(x), dtype=bool)]
        labeled = False
        for mask in groups:
            if not mask.any():
                continue
            ax.plot(x[mask], y[mask], color=color, alpha=.26, linewidth=1)
            ax.plot(x[mask], smooth(y[mask], args.smooth), color=color, linewidth=2,
                    label=None if labeled else label)
            labeled = True
        if len(y) == 1:
            ax.scatter(x, y, color=color, s=18)

    for k in (1, 2):
        ax = fig.add_subplot(grid[0, k-1])
        for enc in ('dinov3_l', 'qwen3vl_4b'):
            series(ax, f's{k}/gan/{enc}/d_loss', COLORS[enc], LABELS[enc])
        ax.set_yscale('log')
        ax.set_title(f'Generation step {k}: discriminator loss')
        ax.set_ylabel('Softplus D loss (log scale)')
        ax.legend(loc='upper right', frameon=False)
        axes.append(ax)
        mmd_row = 2 if has_adv else 1
        if has_adv:
            ax = fig.add_subplot(grid[1, k-1])
            for enc in ('dinov3_l', 'qwen3vl_4b'):
                series(ax, f's{k}/gan/{enc}/g_loss', COLORS[enc], LABELS[enc], active=True)
            ax.set_title(f'Generation step {k}: actual adversarial G loss')
            ax.set_ylabel('Softplus G loss')
            ax.legend(frameon=False)
            axes.append(ax)
        ax = fig.add_subplot(grid[mmd_row, k-1])
        for enc in ('dinov3_l', 'siglip2', 'aimv2_huge'):
            series(ax, f's{k}/{enc}/mmd2', COLORS[enc], LABELS[enc])
        ax.set_title(f'Generation step {k}: MMD')
        ax.set_ylabel('Raw MMD squared')
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.legend(loc='upper right', frameon=False, ncol=3)
        axes.append(ax)

    ax = fig.add_subplot(grid[-1, :])
    series(ax, 'loss_total', '#303843', 'G training loss_total')
    if has_adv:
        series(ax, 'force', '#318564', 'MMD-force contribution')
    ax.set_title('Generator training surrogate (sum over both generation steps)')
    ax.set_ylabel('Training loss')
    ax.legend(loc='upper right', frameon=False)
    if not has_adv:
        ax.text(.025, .07, 'GAN-G is inactive in this snapshot; only MMD force contributes.',
                transform=ax.transAxes, fontsize=10, color='#515c66')
    axes.append(ax)
    for ax in axes:
        ax.set_xlim(1, max(2, int(steps[-1])))
        ax.set_xlabel('Completed training iteration')
        ax.grid(axis='y', alpha=.16, linewidth=.7)
        ax.set_axisbelow(True)
        ax.xaxis.get_major_locator().set_params(integer=True)
        if has_adv:
            ax.axvline(50.5, color='#8a9299', linewidth=1, linestyle='--', alpha=.8)
        for action in data.get('D_lr_actions', []):
            ax.axvline(action['effective_from_step']-.5, color='#9950a2',
                       linewidth=1, linestyle=':', alpha=.8)
        if resumed:
            ax.axvline(200.5, color='#57636c', linewidth=1, linestyle='-.', alpha=.8)
    fig.suptitle(f'Two-step A training losses | iterations 1–{steps[-1]}',
                 x=.09, ha='left', y=.97, fontsize=18, fontweight='medium')
    fig.text(.09, .932, f'Thin lines: every recorded iteration     Solid lines: trailing {args.smooth}-iteration mean',
             fontsize=10, color='#515c66')
    fig.text(.09, .908, '\n'.join(event_lines), fontsize=9, color='#515c66')
    fig.text(.09, .028, 'GAN: DINOv3 + Qwen-VL.  MMD: DINOv3 + SigLIP2 + AIMv2.  GAN-G begins at iteration 51.',
             fontsize=10, color='#515c66')
    lineage = (f"1-200 {INITIAL_SOURCE[:12]}; 201-{steps[-1]} {RESUMED_SOURCE[:12]}"
               if resumed else data['source_commit'][:12])
    fig.text(.09, .009, 'Snapshot '+data['observed_utc']+'  |  training sources '+lineage,
             fontsize=8, color='#68737c')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for ext in ('png', 'pdf'):
        path = args.output.with_suffix('.'+ext)
        fig.savefig(path, dpi=240)
        files.append(str(path.resolve()))
    plt.close(fig)
    print(json.dumps({'last_iteration': int(steps[-1]), 'points': len(rows),
                      'has_actual_GAN_G': has_adv, 'resume200_marked': resumed, 'files': files}))


if __name__ == '__main__':
    main()
