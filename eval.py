from library.evaluation import *
from library.utils import attachment2span, read_attachment_file
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ref", required=True)
parser.add_argument("--pred", required=True)
config = parser.parse_args()

def f1_eval(preds, refs):
    preds_spans = [attachment2span(a) for a in preds]
    refs_spans = [attachment2span(a) for a in refs]
    return f"CorpusF1: {corpus_f1(preds_spans, refs_spans)}"

def acc_eval(preds, refs):
    return f"CorpusUAS: {corpus_acc(preds, refs, add_zeros=True)}"

def eval(preds, refs):
    return f1_eval(preds, refs) + "\t" + acc_eval(preds, refs)

prediction = read_attachment_file(config.pred)
gold = read_attachment_file(config.ref)
print(eval(prediction, gold))
