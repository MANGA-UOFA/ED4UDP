from library.evaluation import corpus_acc
from library.utils import read_attachment_file, flatten
import numpy as np
from library.ensemble import ensemble
import torch
from library.model_selection import *
from references import INDIVIDUALS, GOLD
import time
print(f"GPU is {'' if torch.cuda.is_available() else 'not '}available")

ALPHA_RANGE = np.arange(0.0, 1.01, .1)
USE_DEV_WEIGHTS_FOR_SELECTION = False
USE_DEV_WEIGHTS_FOR_ENSEMBLE = False
ONE_PER_EACH_GROUP = False
POOL_SIZE = 15 # 'all'

TEACHER_PATHS = [[tps[5], tps[2], tps[3], tps[4], tps[1], tps[0]] for tps in INDIVIDUALS]
FINE_TUNING_K = 5

def main(
    METHOD = 'forward society entropy',
    ALPHA = .1,
    SEED = 2,
    MAX_K = None,
    FINE_TUNE_ALPHA = False,
    TIME_ONLY = False,
    PRINT_TIME = True,
    USE_GPU = True,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if USE_GPU else 'cpu'
    teachers_paths = TEACHER_PATHS.copy()
    gold_path = GOLD

    assert METHOD in ALL_METHODS
    assert not ((METHOD not in TUNABLE_METHODS) and (FINE_TUNE_ALPHA))
    assert not ((METHOD not in NONFORWARD_METHODS) and (ONE_PER_EACH_GROUP))
    assert not (POOL_SIZE!='all' and ONE_PER_EACH_GROUP)
    assert not (TIME_ONLY and FINE_TUNE_ALPHA)
    assert not (TIME_ONLY and (MAX_K is None))

    groups_size = len(teachers_paths[0]) if ONE_PER_EACH_GROUP else -1
    teachers_paths = flatten(teachers_paths)
    if POOL_SIZE!='all':
        np.random.seed(SEED)
        teachers_paths = np.random.choice(teachers_paths, size=POOL_SIZE, replace=False).tolist()


    gold_te = read_attachment_file(gold_path.format('te'))
    gold_d = read_attachment_file(gold_path.format('d'))
    preds_te = [read_attachment_file(pred.format('te')) for pred in teachers_paths]
    preds_d = [read_attachment_file(pred.format('d')) for pred in teachers_paths]
    individuals_UAS_d = np.array([corpus_acc(p, gold_d) for p in preds_d])
    cpu_weights = individuals_UAS_d if USE_DEV_WEIGHTS_FOR_SELECTION else [1]*len(preds_d)
    torch_weights = torch.tensor(cpu_weights).to(device)
    ensemble_weights = individuals_UAS_d if USE_DEV_WEIGHTS_FOR_ENSEMBLE else [1]*len(preds_d)
    individuals_UAS_d = torch.from_numpy(individuals_UAS_d).to(device)
    oh_flatten_preds_d = torch.Tensor([flatten(pred) for pred in preds_d]).to(device) # KxN
    oh_flatten_preds_d = torch.nn.functional.one_hot(oh_flatten_preds_d.to(torch.int64), int((oh_flatten_preds_d.max()+1).item())) # KxNxdim


    if FINE_TUNE_ALPHA:
        best_alpha, best_score = -1, -1
        for alpha in ALPHA_RANGE:
            selected_indexes = find_best_combination(
                preds_d,
                gold_d,
                oh_flatten_preds_d,
                individuals_UAS_d,
                FINE_TUNING_K,
                alpha=alpha,
                torch_weights=torch_weights,
                cpu_weights=cpu_weights,
                groups_size=groups_size,
                method=METHOD,
                device=device,
            )
            ensemble_output = ensemble(
                [p for j, p in enumerate(preds_d) if j in selected_indexes],
                weights=[cpu_weights[j] for j in selected_indexes]
            )
            ensemble_UAS = corpus_acc(ensemble_output, gold_d)
            print('Alpha =', alpha, '|', 'UAS:\t', ensemble_UAS)
            if ensemble_UAS > best_score:
                best_alpha, best_score = alpha, ensemble_UAS
        print('-'*25)
        print('Best alpha:', best_alpha)

    else:
        _, selected_indexes = torch.topk(individuals_UAS_d, 1)
        selected_indexes = selected_indexes.tolist()
        k_range = range(groups_size, groups_size+1) if ONE_PER_EACH_GROUP else\
            range(1, len(preds_d)+1 if MAX_K is None else MAX_K+1) if METHOD=='topk' else\
            range(2, len(preds_d) if MAX_K is None else MAX_K+1)
        total_time = 0
        for k in k_range:
            start_time = time.time()
            selected_indexes = find_best_combination(
                preds_d,
                gold_d,
                oh_flatten_preds_d,
                individuals_UAS_d,
                k,
                alpha=ALPHA,
                torch_weights=torch_weights,
                cpu_weights=cpu_weights,
                groups_size=groups_size,
                method=METHOD,
                k_1_selection=selected_indexes,
                device=device,
            )
            end_time = time.time()
            total_time += end_time-start_time
            if not TIME_ONLY:
                ensemble_output = ensemble(
                    [p for j, p in enumerate(preds_te) if j in selected_indexes],
                    weights=[ensemble_weights[j] for j in selected_indexes]
                )
                print('k =', k, '|', 'UAS:\t', corpus_acc(ensemble_output, gold_te))
                print([teachers_paths[j] for j in selected_indexes])
                print('-'*25)
        if PRINT_TIME:
            print(total_time)
        return total_time

main('society entropy', ALPHA=None, FINE_TUNE_ALPHA=True)

times = []
for seed in range(30):
    print("seed:", seed)
    times.append(main('forward society entropy', ALPHA=.4, SEED=seed, MAX_K=5))

print(np.mean(times), np.std(times))