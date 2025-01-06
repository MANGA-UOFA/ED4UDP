import numpy as np

def sent_f1(pred_spans, ref_spans):
    intersection = set(pred_spans).intersection(set(ref_spans))
    return 2 * len(intersection) / (len(pred_spans)+len(ref_spans))

def mean_sent_f1(preds, refs):
    return np.mean([sent_f1(p, r) for p, r in zip(preds, refs)])

def corpus_f1(preds, refs):
    total_intersection, total_pred, total_ref = 0, 0, 0
    for p, r in zip(preds, refs):
        intersection = set(p).intersection(set(r))
        total_intersection += len(intersection)
        total_pred += len(p)
        total_ref += len(r)
    return 2 * total_intersection / (total_pred+total_ref)

def sent_acc(pred_attachments, ref_attachments):
    return np.mean(np.array(pred_attachments)==np.array(ref_attachments))

def mean_sent_acc(preds, refs):
    return np.mean([sent_f1(p, r) for p, r in zip(preds, refs)])

def corpus_acc(preds, refs, add_zeros=False):
    true, total = 0, 0
    for p, r in zip(preds, refs):
        if add_zeros:
            while len(r) > len(p):
                p.append(0)
        true += np.sum(np.array(p)==np.array(r))
        total += len(p)
    return true/total

def alignment(preds1, preds2, refs):
    true_true, false_false, true_false, false_true = 0, 0, 0, 0
    total = 0
    for p1, p2, r in zip(preds1, preds2, refs):
        p1 = np.array(p1)==np.array(r)
        p2 = np.array(p2)==np.array(r)
        true_true += np.sum(np.logical_and(p1, p2))
        false_false += np.sum(np.logical_and(np.logical_not(p1), np.logical_not(p2)))
        true_false += np.sum(np.logical_and(p1, np.logical_not(p2)))
        false_true += np.sum(np.logical_and(np.logical_not(p1), p2))
        total += len(r)
    return {
        'TT': true_true/total,
        'FF': false_false/total,
        'TF': true_false/total,
        'FT': false_true/total
    }