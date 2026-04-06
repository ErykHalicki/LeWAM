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
total_losses = [d['total_loss'] for d in data]
video_losses = [d['video_loss'] for d in data]
action_losses = [d['action_loss'] for d in data]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(steps, total_losses, alpha=0.25, color='steelblue', linewidth=0.8)
ax.plot(steps, video_losses, alpha=0.25, color='green', linewidth=0.8)
ax.plot(steps, action_losses, alpha=0.25, color='orange', linewidth=0.8)
if len(total_losses) >= args.smooth:
    kernel = np.ones(args.smooth) / args.smooth
    smooth_steps = steps[args.smooth // 2 : args.smooth // 2 + len(np.convolve(total_losses, kernel, mode='valid'))]
    for raw, color, name in [
        (total_losses, 'steelblue', 'total'),
        (video_losses, 'green', 'video'),
        (action_losses, 'orange', 'action'),
    ]:
        smoothed = np.convolve(raw, kernel, mode='valid')
        ax.plot(smooth_steps, smoothed, color=color, linewidth=2, label=f'{name} (k={args.smooth})')
    ax.legend()

ax.set_xlabel('step')
ax.set_ylabel('loss')
ax.set_title(args.losses_json)
plt.tight_layout()
plt.show()
