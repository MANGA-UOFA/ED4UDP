from numpy import argmax, inf
from collections import defaultdict
from library.utils import get_length_given_spans

def search(span_scores, length, beta=1): #beta is for using f_{beta} score as the similarity metric
    # l: span length, b: span begin, e: span end, a: adding spans
    # j: breaking point
    scores = [[None for b in range(length-l+1)] for l in range(length+1)]
    spans = [[None for b in range(length-l+1)] for l in range(length+1)]

    def get_score(l, b):
        if l==0:
            return 0
        return scores[l][b]
        
    def get_span(l, b):
        if l==0:
            return []
        return spans[l][b]

    def excl_inner_search(l, b): # search for best j
        scenarios = [] # j, score
        for j in range(1, l):
            scenarios.append((j, get_score(j,b)+get_score(l-j,b+j)))
        return max(scenarios, key=lambda x: x[1]) if scenarios else (None, -inf)

    def incl_inner_search(l, b, this_span_score): # search for best j
        if l==1:
            return (1, this_span_score)
        # j-th word is the head
        scenarios = [] # j, score
        for j in range(1, l+1):
            scenarios.append((j, get_score(j-1,b)+get_score(l-j,b+j)+this_span_score))
        return max(scenarios, key=lambda x: x[1]) if scenarios else (None, -inf)

    for l in range(1, length+1):
        for b in range(length-l+1):
            e = b+l
            this_span = str((b+1, e+1))
            this_span_score = span_scores.get(this_span, 0)

            best_inclusive_j, best_inclusive_score = incl_inner_search(l, b, this_span_score=this_span_score)
            best_exclusive_j, best_exclusive_score = excl_inner_search(l, b) if l > 1 else (None, -inf)
            inclusive = best_inclusive_score >= best_exclusive_score
            best_j, best_score = (best_inclusive_j, best_inclusive_score) \
                if inclusive else (best_exclusive_j, best_exclusive_score)
            best_spans = (get_span(best_j-1, b) + get_span(l-best_j, b+best_j) + [this_span]) \
                if inclusive else (get_span(best_j, b) + get_span(l-best_j, b+best_j))
            scores[l][b] = best_score
            spans[l][b] = best_spans

    return get_span(length, 0), get_score(length, 0)


def sum_span_scores(spans, key_function=str):
    output = defaultdict(lambda: 0)
    for span, score in spans:
        output[key_function(span)] += score
    return output


def nonbinary_in_average(teachers_spans, weights=None, vote_ignoring_level=0, beta=1):
    K = len(teachers_spans)
    n = get_length_given_spans(teachers_spans[0])
    all_spans = []
    for spans, weight in zip(teachers_spans, weights if weights else [1]*K):
        all_spans += [(span, weight) for span in spans]
    span_scores = sum_span_scores(all_spans, key_function=str)
    span_scores = {k: v for k, v in span_scores.items() if v>vote_ignoring_level}
    selected_spans, score = search(span_scores, n, beta=beta)
    selected_spans = [tuple([int(s) for s in span[1:-1].split(', ')]) for span in selected_spans]
    return selected_spans
