from library.evaluation import corpus_acc
from library.utils import flatten, torch_weighted_mean
import numpy as np
from library.ensemble import ensemble
from itertools import combinations, product
from tqdm import tqdm
import torch

TUNABLE_METHODS = [
    'disagreement',
    'KW variance',
    'FK',
    'Kuncheva',
    'PCDM',
    'society entropy',
]
NONFORWARD_METHODS = ['topk']+TUNABLE_METHODS
FORWARD_METHODS = ['forward '+tm for tm in TUNABLE_METHODS]+['ensemble validation']
ALL_METHODS = FORWARD_METHODS+NONFORWARD_METHODS


def disagreement(
    preds, # KxNxD: N samples, K teachers, D classes
    weights, # K
    batch_indexes, # Bxk: B batch size, k to be selected
):
    preds = torch.stack([preds[indexes] for indexes in batch_indexes]).float() # BxkxNxD
    weights = torch.stack([weights[indexes] for indexes in batch_indexes]).unsqueeze(-1).float() # Bxkx1
    preds = preds.unsqueeze(1) - preds.unsqueeze(2) # BxkxkxNxD
    preds = preds.reshape(*preds.shape[:3], -1).unsqueeze(-1) # BxkxkxNDx1
    preds = (preds.transpose(-1, -2) @ preds).squeeze(-1, -2) / preds.shape[-2] # Bxkxk
    probs = weights.transpose(-1, -2) @ preds @ weights # Bx1x1
    probs = probs.reshape(-1) # B
    return probs, weights.squeeze(-1)

def KW_variance(
    preds, # KxNL: K teachers, N samples, L length
    gold, # NL
    weights, # K
    batch_indexes, # Bxk: B batch size, k to be selected
):
    oracle = (preds==gold.unsqueeze(0)).float() # KxNL
    oracle = torch.stack([oracle[indexes] for indexes in batch_indexes]) # BxkxNL
    weights = torch.stack([weights[indexes] for indexes in batch_indexes]) #Bxk
    oracle = oracle*weights.unsqueeze(-1) # BxkxNL
    oracle = oracle.sum(axis=1) # BxNL
    L = weights.sum(axis=-1) # B
    oracle = (oracle)*(L.unsqueeze(-1)-oracle) # BxNL
    oracle = oracle.sum(axis=1) / (oracle.shape[1]*(L**2)) # B
    return oracle, weights

def FK(
    preds, # KxNL: K teachers, N samples, L length
    gold, # NL
    weights, # K
    batch_indexes, # Bxk: B batch size, k to be selected
):
    oracle = (preds==gold.unsqueeze(0)).float() # KxNL
    oracle = torch.stack([oracle[indexes] for indexes in batch_indexes]) # BxkxNL
    weights = torch.stack([weights[indexes] for indexes in batch_indexes]) #Bxk
    oracle = oracle*weights.unsqueeze(-1) # BxkxNL
    p = torch_weighted_mean(oracle, weights).mean(axis=1) # B
    oracle = oracle.sum(axis=1) # BxNL
    L = weights.sum(axis=-1) # B
    oracle = (oracle)*(L.unsqueeze(-1)-oracle) # BxNL
    N = oracle.shape[1]
    oracle = oracle.sum(axis=1) / (L*N*(L-1)*p*(1-p)) # B
    return oracle, weights

def PCDM(
    preds, # KxNL: K teachers, N samples, L length
    gold, # NL
    weights, # K
    batch_indexes, # Bxk: B batch size, k to be selected
    T_low=.1,
    T_high=.9,
):
    oracle = (preds==gold.unsqueeze(0)).float() # KxNL
    oracle = torch.stack([oracle[indexes] for indexes in batch_indexes]) # BxkxNL
    weights = torch.stack([weights[indexes] for indexes in batch_indexes]) #Bxk
    oracle = torch_weighted_mean(oracle, weights) # BxNL
    oracle = torch.logical_and(oracle>=T_low, oracle<=T_high).float().mean(axis=1) #B
    return oracle, weights

def Kuncheva(
    preds, # KxNL: K teachers, N samples, L length
    gold, # NL
    weights, # K
    batch_indexes, # Bxk: B batch size, k to be selected
):
    oracle = (preds==gold.unsqueeze(0)).float() # KxNL
    oracle = torch.stack([oracle[indexes] for indexes in batch_indexes]) # BxkxNL
    weights = torch.stack([weights[indexes] for indexes in batch_indexes]) #Bxk
    oracle = oracle*weights.unsqueeze(-1) # BxkxNL
    oracle = oracle.sum(axis=1) # BxNL
    L = weights.sum(axis=-1) # B
    oracle = torch.minimum(oracle, L.unsqueeze(-1)-oracle) / torch.ceil(L/2).unsqueeze(-1) # BxNL
    return oracle.mean(axis=1), weights

def society_entropy(
    preds, # KxNxD: N samples, K teachers, D classes
    weights, # K
    batch_indexes, # Bxk: B batch size, k to be selected
):
    preds = torch.stack([preds[indexes] for indexes in batch_indexes])
    weights = torch.stack([weights[indexes] for indexes in batch_indexes])
    probs = torch_weighted_mean(preds, weights)
    entropy = -torch.nansum(probs*torch.log(probs), axis=2)
    return entropy.mean(axis=1), weights
    

def find_best_combination(
    preds,
    gold,
    oh_oh_flatten_preds,
    individuals_UAS,
    k,
    alpha,
    cpu_weights,
    torch_weights,
    device='cpu',
    k_1_selection=[],
    batch_ratio=20000,
    groups_size=-1,
    method='topk',
):
    if method=='topk':
        _, best_indexes = torch.topk(individuals_UAS, k)
        return best_indexes.tolist()
    
    if method=='ensemble validation':
        k_1_selection = list(k_1_selection)
        best_i, best_UAS = -1, -1
        for i in tqdm(range(len(preds))):
            if i not in k_1_selection:
                new_selection = k_1_selection+[i]
                ensemble_output = ensemble(
                    [p for j, p in enumerate(preds) if j in new_selection],
                    weights=[cpu_weights[j] for j in new_selection],
                    progress_bar=False
                )
                ensemble_UAS = corpus_acc(ensemble_output, gold)
                if ensemble_UAS > best_UAS:
                    best_i, best_UAS = i, ensemble_UAS
        return k_1_selection+[best_i]
    
    else:
        all_indexes = \
            [list(k_1_selection)+[i] for i in range(len(preds)) if i not in k_1_selection] \
            if method in FORWARD_METHODS else \
            list(map(list, combinations(list(range(len(preds))), k))) \
            if groups_size<0 else \
            list(map(list, product(*[np.arange(0, len(preds), groups_size)+i for i in range(groups_size)])))
        batch = batch_ratio // k
        if 'disagreement' in method:
            batch = int(batch // (k**.25))

        all_scores = []
        for b in tqdm(range(0, len(all_indexes), batch)):
            batch_indexes = all_indexes[b:b+batch]
            
            if 'society entropy' in method:
                scores, batch_weights = society_entropy(oh_oh_flatten_preds, torch_weights, batch_indexes)
            elif 'disagreement' in method:
                scores, batch_weights = disagreement(oh_oh_flatten_preds, torch_weights, batch_indexes)
            elif 'KW variance' in method:
                scores, batch_weights = KW_variance(
                    torch.Tensor([flatten(pred) for pred in preds]).to(device),
                    torch.Tensor(flatten(gold)).to(device),
                    torch_weights,
                    batch_indexes
                )
            elif 'FK' in method:
                scores, batch_weights = FK(
                    torch.Tensor([flatten(pred) for pred in preds]).to(device),
                    torch.Tensor(flatten(gold)).to(device),
                    torch_weights,
                    batch_indexes
                )
            elif 'Kuncheva' in method:
                scores, batch_weights = Kuncheva(
                    torch.Tensor([flatten(pred) for pred in preds]).to(device),
                    torch.Tensor(flatten(gold)).to(device),
                    torch_weights,
                    batch_indexes
                )
            elif 'PCDM' in method:
                scores, batch_weights = PCDM(
                    torch.Tensor([flatten(pred) for pred in preds]).to(device),
                    torch.Tensor(flatten(gold)).to(device),
                    torch_weights,
                    batch_indexes
                )

            batch_individuals_UAS = torch_weighted_mean(torch.stack([individuals_UAS[indexes] for indexes in batch_indexes]), batch_weights)
            scores = batch_individuals_UAS + alpha*scores

            all_scores += scores.tolist()

        return all_indexes[np.argmax(all_scores)]