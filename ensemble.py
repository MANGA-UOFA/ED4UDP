from library.evaluation import *
from library.ensemble import ensemble
from library.utils import read_attachment_file
from references import INDIVIDUALS, GOLD

FOLD = 'te'
AGG = 'acc'
USE_DEV_WEIGHTS = False

teachers_paths = [[tps[5], tps[2], tps[3], tps[4], tps[1], tps[0]] for tps in INDIVIDUALS] #Sorted
output_path = "outputs/"+FOLD+"_{}.txt"

preds = [[read_attachment_file(t.format(FOLD)) for t in tps] for tps in teachers_paths]
gold = read_attachment_file(GOLD.format(FOLD))

if USE_DEV_WEIGHTS:
    preds_d = [[read_attachment_file(t.format('d')) for t in tps] for tps in teachers_paths]
    gold_d = read_attachment_file(GOLD.format('d'))
    avg = [ensemble(pr, agg=AGG, parallel=True, weights=[corpus_acc(pds, gold_d) for pds in pr_d]) for pr, pr_d in zip(preds, preds_d)]
else:
    avg = [ensemble(pr, agg=AGG, parallel=True) for pr in preds]

for i, ag in enumerate(avg):
    with open(output_path.format(i+1), 'w') as f:
        out = '\n'.join([' '.join(map(str, [j]+a)) for j,a in enumerate(ag)])
        f.write(out)