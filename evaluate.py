import os
import json
import numpy as np
import pandas as pd
import re
import math
import language_tool_python as lt
from collections import Counter
import matplotlib
import re

# import spacy
# from spacy_readability import Readability


# from torchtext.data.metrics import bleu_score
from metrics import correctness, edit_distance

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



def summary_paragraph(text):
    length = 0
    sentences = text.split(".")
    for s in sentences:
        length += len(s.split())
    return length, len(sentences)

num_selected = np.zeros((1,3))
num_drafts = np.zeros((1,3))
num_paras = np.zeros((1,3))
total = np.zeros((1,3))
#all section len
len_before_para = np.zeros((1,3))
len_after_para = np.zeros((1,3))
num_sen_before = np.zeros((1,3))
num_sen_after = np.zeros((1,3))

#edited sentence len
len_before = np.zeros((1,3))
len_after = np.zeros((1,3))



delta = np.zeros((1,3))
edit_dis = np.zeros((1,3))
bleu = np.zeros((1,3))
num_pair = np.zeros((1,3))
correct = np.zeros((1,3))

#cleaned up edits
clean_edits = [[],[],[]]
reg_set = "^[ A-Za-z0-9,.!%^&*()?/|:;_-]*$"

years = [2018, 2019, 2020, 2021]
paras = ['abstract','introduction','conclusion']
for year in years:
    print(year)
    if year == 2018:
        sections = ['accepted-oral-papers','accepted-poster-papers','rejected-papers','workshop-papers']
    elif year == 2019:
        sections = ['accepted-oral-papers','accepted-poster-papers','rejected-papers']
    elif year == 2020:
        sections = ['accept-spotlight','accept-talk','accept-poster','reject']
    elif year == 2021:
        sections = ['oral-presentations','spotlight-presentations','poster-presentations','withdrawn-rejected-submissions']

    for section in sections:
        if 'oral' in section or 'spotlight' in section:
            index = 0
        elif 'poster' in section:
            index = 1
        else:
            index = 2
        prefix = f"open_review/ICLR{year}"
        id_path = f'{prefix}/raw/{section}.txt'
        output_path = f'open_review/processed/ICLR{year}/{section}'

        paper_list = []
        with open(id_path, 'r') as f:
            paper_list = f.read().splitlines() 
            total[0,index] += len(paper_list)
        i = 0
        for paper in paper_list:
#             if i>0:
#                 break
            print(i)
            path = output_path + '/' + paper + '.json'
            if os.path.exists(path):
                num_drafts[0,index]+=1
            else:
                continue
            with open(path, 'r') as f:
                data = json.load(f)
            if 'edits' not in data:
                continue
            pairs = data['edits']
            for para_name in paras:
                if para_name in data:
                    para = data[para_name]
                    if len(para)!=2:
                        continue
                    num_paras[0,index]+=1
                    content_before, content_after = para
#                     doc = nlp(content_before)
                    a, b = summary_paragraph(content_before)
                    c, d = summary_paragraph(content_after)
                    len_before_para[0,index]+= a
                    num_sen_before[0,index]+=b
                    len_after_para[0,index]+=c
                    num_sen_after[0,index]+=d
                    correct_score = correctness(content_before)
                    print(correct_score)
                    correct[0,index] += correct_score
                    
#                     print(doc._.flesch_kincaid_grade_level)
#                     print(doc._.flesch_kincaid_reading_ease)
#                     print(doc._.dale_chall)
#                     print(doc._.smog)
#                     print(doc._.coleman_liau_index)
#                     print(doc._.automated_readability_index)
#                     print(doc._.forcast)
            flag = 0
            for sec_edits in pairs:
                for s in sec_edits:
                    check_pre = re.match(reg_set, s[0])
                    check_after = re.match(reg_set, s[1])
                    if check_pre and check_after:
                        dis = edit_distance(s[0],s[1])
                        if len(s[0])!=0 and len(s[1])!=0:
                            edit_dis[0,index]+=dis
    #                         if dis<=15:
                            flag = 1
                            num_pair[0,index]+=1
                            bleu[0,index]+=bleu_score(s[0].split(),s[1].split())
                            len_before[0,index]+= len(s[0].split())
                            len_after[0,index]+= len(s[1].split())
                            delta[0,index]+=(len(s[1].split())-len(s[0].split()))
                            clean_edits[index].append(s)
            num_selected[0,index]+=flag
            i+=1


    print("===================")
    print("total")
    print(total)
    print(num_drafts)
    print(num_selected)
    print("num of pairs")
    print(num_pair)
print("=====================")
print("paragraph level stats")
print(num_sen_before/num_drafts)
print(num_sen_after/num_drafts)
print(len_before_para/num_sen_before)
print(len_after_para/num_sen_after)
print("correctness")
print(correct/num_paras)


print("===================")
print("sentence level stats")
print(len_before/num_pair)
print(len_after/num_pair)
print(delta/num_pair)
print(edit_dis/num_pair)
# print(clean_edits)

for i in range(3):
    with open(f'clean_edits_{i}.tsv', 'w') as f:
        for item in clean_edits[i]:
            f.write("%s" % item[0])
            f.write("\t\t")
            f.write("%s" % item[1])
            f.write("\n")