import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

_default = os.path.join(os.environ.get('LE_WAM_ROOT', '.'), 'runs', 'recent', 'losses.json')

parser = argparse.ArgumentParser()
parser.add_argument('losses_json', nargs='?', default=_default)
parser.add_argument('--smooth', type=int, default=50)
args = parser.parse_args()

with open(args.losses_json) as f:
    data = json.load(f)

steps  = [d['step'] for d in data]
losses = [d['loss'] for d in data]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(steps, losses, alpha=0.25, color='steelblue', linewidth=0.8)

if len(losses) >= args.smooth:
    kernel = np.ones(args.smooth) / args.smooth
    smoothed = np.convolve(losses, kernel, mode='valid')
    smooth_steps = steps[args.smooth // 2 : args.smooth // 2 + len(smoothed)]
    ax.plot(smooth_steps, smoothed, color='steelblue', linewidth=2, label=f'smoothed (k={args.smooth})')
    ax.legend()

ax.set_xlabel('step')
ax.set_ylabel('loss')
ax.set_title(args.losses_json)
plt.tight_layout()
plt.show()
