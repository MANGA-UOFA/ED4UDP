import pandas as pd
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/local/ssd_1/behzad/EUDP/')
import seaborn as sns
from collections import defaultdict
from library.utils import read_attachment_file, flatten
from references import INDIVIDUALS, GOLD
ENSEMBLE_PATH = ["outputs/ensemble/{}_"+str(i)+".txt" for i in range(1,6)]
GOLD_TAGS = "outputs/Gold/{}_pos_tags.txt"

FOLD = 'te'
teachers_paths = [[tps[5], tps[2], tps[3], tps[4], tps[1], tps[0], tps[6]][1:-1] for tps in INDIVIDUALS]

preds = [[flatten(read_attachment_file(t.format(FOLD))) for t in tps] for tps in teachers_paths]
gold = flatten(read_attachment_file(GOLD.format(FOLD)))
gold_tags = flatten(read_attachment_file(GOLD_TAGS.format(FOLD), transform=str))
ensemble = [flatten(read_attachment_file(t.format(FOLD))) for t in ENSEMBLE_PATH]
refs = [p+[e] for p,e in zip(preds, ensemble)]

def compute_recall(preds, golds, golds_tags):
    true = defaultdict(lambda: 0)
    all = defaultdict(lambda: 0)
    for k, (g, p, t) in enumerate(zip(golds, preds, golds_tags)):
        all[t] += 1
        true[t] += p==g
    recall = {tag: true[tag]/all[tag] for tag in all}
    total = sum(all.values())
    coverage = {tag: all[tag]/total for tag in all}
    return recall, coverage

def create_recall_df(predictions, golds, labels, name, top=10):
    recall, coverage = compute_recall(
        predictions,
        golds,
        labels,
    )
    recall_df = pd.DataFrame({'tag': recall.keys(), 'Recall': recall.values(), 'Coverage': coverage.values()})
    recall_df = recall_df.sort_values('Coverage', ascending=False).head(top)
    coverage = recall_df['Coverage'].sum()
    recall_df = recall_df.drop(columns='Coverage')
    recall_df['model'] = name
    recall_df['Recall'] *= 100
    return recall_df, coverage*100


model_names = [teachers_paths[0][i].split('/')[-3] for i in range(len(teachers_paths[0]))]+['Ensemble']
model_names = ['Sib-NDMV','L-NDMV','CRFAE','NE-DMV','NDMV','Ensemble']
def figure_4():
    data = []
    for run in range(len(refs)):
        for model in range(len(refs[0])):
            recall_df, coverage = create_recall_df(refs[run][model], gold, gold_tags, model_names[model])
            data.append(recall_df)
    data = pd.concat(data)
    print(f"Coverage of reported tags: {coverage}%")

    y_low = 0
    grouped_data = data.groupby(['tag','model'])['Recall']
    y_max = (grouped_data.mean()+grouped_data.std()).max()
    fig, ax = plt.subplots(figsize=(15, 4))
    ax = sns.barplot(data=data, x="tag", y="Recall", hue="model", palette=sns.cubehelix_palette(5, reverse=True)+['#17A3BF'],
                    err_kws={'linewidth': 1, 'color': 'gray'}, capsize=0.05, edgecolor='k', errorbar="sd", width=0.9, alpha=.45)
    for i, bars in enumerate(ax.containers):
        for j, bar in enumerate(bars):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                y_low+2,
                model_names[i],
                ha='center',
                va='bottom',
                color='black',
                fontsize=10,
                rotation=90,
            )
    plt.legend().set_visible(False)
    plt.xlabel('').set_visible(False)
    plt.ylabel(r'Recall on PTB test', fontsize=13).set_visible(False)
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    plt.ylim(y_low, y_max)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig('plots/recalls.pdf', bbox_inches='tight')

figure_4()