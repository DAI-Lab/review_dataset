import math
from collections import Counter

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from spacy_readability import Readability
# from spacy_grammar.grammar import Grammar

import spacy
import nltk
from spacy.tokens import Token


def paragraph_summary_statistics(tool, sentences):
    total_tokens = 0
    num_sen = len(sentences)
    total_tokens = np.sum([len(s.split()) for s in sentences])
    output = {}
    nlp = spacy.load('en_core_web_sm')

    output['readability'] = compute_readability(nlp, sentences)
    output['correctness'] = compute_correctness(tool, sentences)
    # output['convincing'] = compute_convincing(nlp, sentences)
    output['num_sen'] = num_sen
    if num_sen == 0:
        output['avg_sen_len'] = 0
    else:
        output['avg_sen_len'] = total_tokens/num_sen
    return output

def compute_correctness(tool, sentences):
    count_errors = 0
    for s in sentences:
        matches = tool.check(s)
        if len(matches)>1:
            count_errors+=1
    if len(sentences) == 0:
        score = 1.0
    else:
        score = 1.0-(count_errors/len(sentences))
    return score


def compute_readability(nlp, sentences):
    read = Readability()
    nlp.add_pipe(read, last=True)

    scores = []
    if len(sentences) == 0:
        return 0.0
    # Token.set_extension('context', default=False, force=True)
    for s in sentences:
        sent = nlp(s)
        avg_score = sent._.flesch_kincaid_grade_level + sent._.coleman_liau_index
                    # sent._.automated_readability_index + \
        if not avg_score:
            scores.append(0)
        else:
            scores.append(avg_score/3)
    return np.mean(scores)


def compute_convincing(nlp, sentences):
    tagger = nlp.add_pipe("tagger")
    if len(sentences)==0:
        return 1
    for s in sentences: 
        sent = nlp(s)
        tags = [token.pos_ for token in s]
        print(tags)
    return 0



def edit_distance(before, after):
    before = before.split()
    after = after.split()
    if len(before) == 0 or len(after) == 0:
        return len(before) + len(after)

    f = np.zeros((len(before) + 1, len(after) + 1), dtype='int')
    for i in range(len(before)):
        for j in range(len(after)):
            f[i + 1][j + 1] = min(f[i, j + 1] + 1, f[i + 1, j] + 1, f[i, j] + 1)
            if before[i] == after[j]:
                f[i + 1][j + 1] = min(f[i + 1][j + 1], f[i][j])

    return int(f[len(before)][len(after)])


def bleu_score(hyp, ref):
    # get ngram stats
    stats = []
    stats.append(len(hyp))
    stats.append(len(ref))
    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hyp[i:i + n]) for i in range(len(hyp) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(ref[i:i + n]) for i in range(len(ref) + 1 - n)]
        )
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hyp) + 1 - n, 0]))

    # get bleu from stats
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    bleu = math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)

    return bleu



def compute_metrics(preds, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    print(preds)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }