import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_style('ticks')

MAIN = True

def read_file(path):
    with open(path) as f:
        out = [
        float(line.strip().split()[-1])
        for line in f.readlines()[1::2]
        ]
    if 'topk' in path:
        out = out[1:-1]
    return out

methods = [
    ['w/o diversity', '#E1C999', 'o', (1,1)],
    ['Ensemble validation', 'gray', 'p', (1,2)],
    ['w/ Kuncheva’s diversity', '#CE88B2', '^', ''],
    ['w/ society entropy', '#4CE0D2', 'P', ''],
] if MAIN else [
    ['w/ disagreement', 'gray', '*', (1,1)],
    ['w/ KW variance', '#E1C999', 'o', (3,2)],
    ['w/ Fleiss’ Kappa', '#226BAB', 'v', ''],
    ['w/ PCDM', '#CE88B2', 'p', ''],
    ['_Hidden w/ society entropy', '#4CE0D2', 'P', ''],
]
names, palette, markers, dashes = zip(*methods)

experiment = 'SOME_LOGS_PATH/{}.log'.format

data = pd.DataFrame({name: read_file(experiment(name))[1:] for name in names})
data.iloc[:, :] *= 100
data.index = list(range(3, len(data)+3))

plt.figure(figsize=(16, 8))
ax = sns.lineplot(
    data,
    alpha=.6,
    dashes=dashes,
    linewidth=4.5,
    palette=palette,
    legend=False,
)
sns.lineplot(
    data,
    markers=['o']*len(markers),
    markersize=20,
    alpha=1,
    linestyle='',
    dashes=False,
    linewidth=3,
    palette=['white']*(len(palette)),
    legend=False,
)
ax = sns.lineplot(
    data,
    markers=markers,
    markersize=14,
    alpha=1,
    linestyle='',
    dashes=False,
    linewidth=3,
    palette=palette,
)

plt.ylim(63.5 + 0.1*(1-MAIN), 68.8 + 0.1*(1-MAIN))
plt.xlim(2.7, len(data)+2.3)
fontsize=24
if MAIN:
    plt.xticks([])
    plt.legend(fontsize=fontsize*1.4)
else:
    plt.xticks(range(3, len(data)+3), fontsize=fontsize)
    plt.xlabel('# ensemble components', fontsize=fontsize*1.4)
    plt.legend(fontsize=fontsize*1.4, ncol=2)
plt.yticks(fontsize=fontsize)
plt.ylabel('Ensemble UAS', fontsize=fontsize*1.4)

props = dict(boxstyle='square', facecolor='white', pad=.29, edgecolor='black')
ax.text(.0105, 1-.078, '(a)' if MAIN else '(b)', transform=ax.transAxes, fontsize=fontsize*1.4, bbox=props)

plt.savefig(open(f'plots/model_selection_{"main" if MAIN else "appndx"}.pdf', 'wb'), bbox_inches='tight')