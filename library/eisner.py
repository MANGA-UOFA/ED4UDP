import numpy as np

L, R = 0, 1
I, C = 0, 1
DIRECTIONS = (L, R)
COMPLETENESS = (I, C)
NEG_INF = -float('inf')


class Span(object):
    def __init__(self, left_idx, right_idx, head_side, complete):
        self.data = (left_idx, right_idx, head_side, complete)

    @property
    def left_idx(self):
        return self.data[0]

    @property
    def right_idx(self):
        return self.data[1]

    @property
    def head_side(self):
        return self.data[2]

    @property
    def complete(self):
        return self.data[3]

    def __str__(self):
        return "({}, {}, {}, {})".format(
            self.left_idx,
            self.right_idx,
            "L" if self.head_side == L else "R",
            "C" if self.complete == C else "I",
        )

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.data)

    def __eq__(self, other):
        return isinstance(other, Span) and hash(other) == hash(self)

def eisner(weight):
    """
    `N` denotes the length of sentence.

    :param weight: size N x N
    :return: the projective tree with maximum score
    """
    N = weight.shape[0]
    if N==2:
        return [(0, 1)]

    btp = {}  # Back-track pointer
    dp_s = {}

    # Init
    for i in range(N):
        for j in range(i + 1, N):
            for dir in DIRECTIONS:
                for comp in COMPLETENESS:
                    dp_s[Span(i, j, dir, comp)] = NEG_INF

    # base case
    for i in range(N):
        for dir in DIRECTIONS:
            dp_s[Span(i, i, dir, C)] = 0.
            btp[Span(i, i, dir, C)] = None

    rules = [
        # span_shape_tuple := (span_direction, span_completeness),
        # rule := (span_shape, (left_subspan_shape, right_subspan_shape))
        ((L, I), ((R, C), (L, C))),
        ((R, I), ((R, C), (L, C))),
        ((L, C), ((L, C), (L, I))),
        ((R, C), ((R, I), (R, C))),
    ]

    for size in range(1, N):
        for i in range(0, N - size):
            j = i + size
            for rule in rules:
                ((dir, comp), ((l_dir, l_comp), (r_dir, r_comp))) = rule

                if comp == I:
                    edge_w = weight[i, j] if (dir == R) else weight[j, i]
                    k_start, k_end = (i, j)
                    offset = 1
                else:
                    edge_w = 0.
                    k_start, k_end = (i + 1, j + 1) if dir == R else (i, j)
                    offset = 0

                span = Span(i, j, dir, comp)
                for k in range(k_start, k_end):
                    l_span = Span(i, k, l_dir, l_comp)
                    r_span = Span(k + offset, j, r_dir, r_comp)
                    s = edge_w + dp_s[l_span] + dp_s[r_span]
                    if s > dp_s[span]:
                        dp_s[span] = s
                        btp[span] = (l_span, r_span)

    # recover tree
    return back_track(btp, Span(0, N - 1, R, C), set())


def back_track(btp, span, edge_set):
    if span.complete == I:
        if span.head_side == L:
            edge = (span.right_idx, span.left_idx)
        else:
            edge = (span.left_idx, span.right_idx)
        edge_set.add(edge)

    if btp[span] is not None:
        l_span, r_span = btp[span]

        back_track(btp, l_span, edge_set)
        back_track(btp, r_span, edge_set)
    else:
        return

    return edge_set

def edges_to_attachment(edge_set):
    edge_set = sorted((edge_set), key=lambda x: x[1])
    edge_set, _ = zip(*edge_set)
    return list(edge_set)


def average(teachers_attachments, weights=None):
    K = len(teachers_attachments)
    if weights is None:
        weights = [1]*K
    count = len(teachers_attachments[0])
    sent = np.zeros((count+1, count+1))
    for j in range(count):
        for k in range(K):
            sent[teachers_attachments[k][j]][j+1] += weights[k]
    sent = np.array(sent)
    sent[0, :] -= 200
    sent[:, 0] = -np.inf
    avg = edges_to_attachment(eisner(sent))
    return avg