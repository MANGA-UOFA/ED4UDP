def get_length_given_spans(spans):
    return max([i[1] for i in spans])-1


class PSTnode: #PhraseStructureTree
    def __init__(self, index):
        self.index = index
        self.children = []

    def get_spans(self):
        if not self.children:
            return [(self.index, self.index+1)]
        spans = sum([c.get_spans() for c in self.children], [])
        left_most = min([span[0] for span in spans]+[self.index])
        right_most = max([span[1] for span in spans]+[self.index+1])
        spans.append((left_most, right_most))
        return spans

# attachment: [2, 0, 2, 5, 3, 7, 8, 9, 5]; j = i-th number shows i-th word is dependent to the j-th word; 0 shows the root
# dependents: [[2], [], [1, 3], [5], [], [4, 9], [], [6], [7], [8]]

def attachment2dependents(attachment):
    dependents = [[] for i in range(len(attachment)+1)]
    for i, a in enumerate(attachment):
        dependents[a].append(i+1)
    return dependents

def dependents2PST(dependents, root=0):
    if root==0:
        root = dependents[0][0]
    node = PSTnode(root)
    for dependent in dependents[root]:
        node.children.append(dependents2PST(dependents, dependent))
    return node

def attachment2PST(attachment):
    return dependents2PST(attachment2dependents(attachment))

def attachment2span(attachment):
    return attachment2PST(attachment).get_spans()

def span2PST(spans):
    def find_head(span):
        for i in range(*span.index):
            for child in span.children:
                if i in range(*child.index):
                    break
            else:
                return i

    length = get_length_given_spans(spans)
    spans = sorted(spans, key=lambda x: x[1]-x[0])
    spans = [PSTnode(s) for s in spans]
    for i, inner in enumerate(spans[:-1]):
        for outter in spans[i+1:]:
            if inner.index[0]>=outter.index[0] and inner.index[1]<=outter.index[1]:
                outter.children.append(inner)
                break
        else:
            print('no parent for:', inner.index)
    for span in reversed(spans):
        span.index = find_head(span)
    return spans[-1], length

def PST2attachment(root, length):
    def set_attachment(node, parent, attachment):
        attachment[node.index-1] = parent
        for child in node.children:
            set_attachment(child, node.index, attachment)

    attachment = [-1]*length
    set_attachment(root, 0, attachment)
    return attachment

def span2attachment(spans):
    return PST2attachment(*span2PST(spans))

def read_attachment_file(path, id=True, transform=int):
    return [[transform(c) for c in line.split()[int(id):]] for line in open(path).readlines()]



def flatten(lol):
    output = []
    for l in lol:
        output += l
    return output

def torch_weighted_mean(s, w, axis=1):
    while len(w.shape)<len(s.shape):
        w = w.unsqueeze(-1)
    s = s * w
    s = s.sum(axis=axis)/w.sum(axis=axis)
    return s