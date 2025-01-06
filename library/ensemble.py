from tqdm import tqdm
import concurrent.futures
from library.f1_agg import nonbinary_in_average as f1_average
from library.eisner import average as acc_average
from library.utils import attachment2span, span2attachment
import time


def f_worker(i, ts, weights, beta):
    ts = [attachment2span(a) for a in ts]
    avg = f1_average(ts,
        weights=weights,
        beta=beta,
    )
    avg = span2attachment(avg)
    return i, avg

def acc_worker(i, ts, weights, beta):
    avg = acc_average(ts,
        weights=weights,
    )
    return i, avg

def ensemble(references, agg='acc', beta=1, weights=None, parallel=True, return_times=False, progress_bar=True):
    worker = acc_worker if agg=='acc' else f_worker
    if weights is None:
        weights = [1]*len(references)
    assert len(weights)==len(references)
    assert not (parallel and return_times)
    bar = tqdm if progress_bar else lambda x, **args: x
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(worker, i, [tree[i] for tree in references], weights, beta) for i in range(len(references[0]))]
            avgs = [f.result() for f in bar(concurrent.futures.as_completed(futures), total=len(futures))]
            avgs = sorted(avgs, key=lambda x: x[0])
            avgs = list(map(lambda x: x[1], avgs))
    else:
        avgs = []
        times = []
        for i in bar(range(len(references[0]))):
            start = time.time()
            a = worker(i, [tree[i] for tree in references], weights, beta)[1]
            end = time.time()
            times.append(end-start)
            avgs.append(a)
    if return_times:
        return avgs, times
    return avgs