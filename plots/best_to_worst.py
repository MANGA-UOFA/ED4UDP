import matplotlib.pyplot as plt
from matplotlib_dashboard import MatplotlibDashboard as MD

n = 6

performance = ['67.9±0.1', '67.7±0.2', '68.5±0.2', '68.5±0.3', '68.5±0.3', '67.9±0.3', '67.9±0.6'][:n]
individuals = ['67.9±0.1', '64.3±0.2', '62.4±0.6', '53.0±4.5', '51.0±3.4', '48.1±3.6', '41.5±0.8'][:n]

means, stds = zip(*[(float(v) for v in p.split('±')) for p in performance])
indmeans, indstds = zip(*[(float(v) for v in p.split('±')) for p in individuals])


fig = plt.figure(figsize=(13,4))
dashboard = MD([
    ['left' ,'right'],
], wspace=0.1)

def plot(ax, base_ax, means, stds, bases, base_stds, color, marker):
    x = list(range(len(means)))
    ax.plot(x, means, linestyle=':', color=color)
    ax.errorbar(x, means, fmt='none', yerr=stds, capsize=7, elinewidth=2.5, capthick=1.5, color=color)
    style = ax.scatter(x, means, marker=marker, s=120, color=color)
    base_ax.errorbar(x, bases, fmt='none', yerr=base_stds, capsize=7, elinewidth=2.5, capthick=1.5, color='#484848', linewidth=0)
    base_style = base_ax.scatter(x, bases, marker=marker, s=120, color='#484848', linewidth=0)
    return ax, base_ax, style, base_style

ax, base_ax, style, base_style = plot(dashboard['right'], dashboard['left'], means, stds, indmeans, indstds, color='black', marker='s')
plt.tight_layout()
ax.yaxis.tick_right()
ax.set_xticks([])
ax.tick_params(axis='y', labelsize=20)
base_ax.tick_params(axis='y', labelsize=20)
base_ax.set_xticks([])

props = dict(boxstyle='square', facecolor='white', pad=.29)
ax.text(.018, .048, '(b)', transform=ax.transAxes, fontsize=22, bbox=props)
base_ax.text(.019, .048, '(a)', transform=base_ax.transAxes, fontsize=22, bbox=props)

labels = ['Sib&L-NDMV', 'Sib-NDMV', 'L-NDMV', 'CRFAE', 'NE-DMV', 'NDMV', 'Flow'][:n]
aligns = ['bottom', 'bottom', 'top', 'top', 'top', 'bottom', 'bottom'][:n]
base_aligns = ['top', 'top', 'top', 'bottom', 'bottom', 'bottom', 'bottom'][:n]

for i, (label, a, ba) in enumerate(zip(labels, aligns, base_aligns)):
    x = means[i]-stds[i]-.05 if a=='top' else means[i]+stds[i]+.06
    ax.text(i, x, label, ha='center', va=a, color='black', fontsize=16, rotation=90)
    base_x = indmeans[i]-indstds[i]-1 if ba=='top' else indmeans[i]+indstds[i]+1
    base_ax.text(i, base_x, label, ha='center', va=ba, color='black', fontsize=16, rotation=90)

plt.savefig('plots/best_to_worst.png', bbox_inches='tight')