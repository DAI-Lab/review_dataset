
import os
import sys
import json
import numpy as np
import pandas as pd
import re
import math
import random
import language_tool_python as lt
from collections import Counter
import matplotlib
from tqdm import tqdm
import language_tool_python
import Levenshtein


# from torchtext.data.metrics import bleu_score
from metrics import edit_distance, bleu_score, paragraph_summary_statistics
from get_data2 import build_tokenizer, clean_up_edits


def filter_sentence(sentences, reg_set):
    filtered = []
    stop_words = ['Figure ', 'Under review as', 'Published as', 'Table ','https']
    for s in sentences:
        check = re.match(reg_set, s)
        flag = True
        if check:
            for w in stop_words:
                if w in s:
                    flag = False
            if flag:         
                filtered.append(s)
    return filtered

if __name__=='__main__':
    
    # sys.setdefaultencoding('utf-8')

    num_selected = np.zeros((4,3))
    num_drafts = np.zeros((4,3))
    num_paras = np.zeros((1,3))
    total = np.zeros((4,3))
    #all section len
    num_sen = np.zeros((2,3))
    avg_sen_len = np.zeros((2,3))
    readability_score = np.zeros((2,3))
    correctness_score = np.zeros((2,3))
    convincing_score = np.zeros((2,3))
    sentiment_score = np.zeros((2,3))
    read_scores = []
    correct_score = []
    papers = []
    alltexts = []
    results = []

    #edited sentence len
    edit_len = np.zeros((2,3))

    delta = np.zeros((1,3))
    edit_dis = np.zeros((1,3))
    bleu = np.zeros((1,3))
    num_pair = np.zeros((4,3))


    tokenizer = build_tokenizer()
    tool = language_tool_python.LanguageTool('en-US')

    reg_set = "^[ A-Za-z0-9,.!%^&*()?/|:;_-]*$"
    stop_words = ['Figure ', 'Under review as', 'Published as', 'Table ','https']
    ids = [[],[],[]]

    years = [2018, 2019, 2020, 2021]
    paras = ['abstract','introduction','conclusion']
    for y in range(len(years)):
        year = years[y]
        if year == 2018:
            # sections = ['accepted-oral-papers']
            sections = ['accepted-oral-papers','accepted-poster-papers','rejected-papers','workshop-papers']
        elif year == 2019:
            sections = ['accepted-oral-papers','accepted-poster-papers','rejected-papers']
        elif year == 2020:
            sections = ['accept-spotlight','accept-talk','accept-poster','reject']
        elif year == 2021:
            sections = ['oral-presentations','spotlight-presentations','poster-presentations','withdrawn-rejected-submissions']

        for section in sections:
            if 'oral' in section or 'spotlight' in section or 'talk' in section:
                index = 0
            elif 'poster' in section or 'workshop' in section:
                index = 1
            else:
                index = 2
            prefix = f"open_review/ICLR{year}"
            id_path = f'{prefix}/raw/{section}.txt'
            data_path = f'{prefix}/data/{section}'
            output_path = f'open_review/new_processed/ICLR{year}/{section}'

            with open(id_path, 'r') as f:
                paper_list = list(set(f.read().splitlines()))
                total[y,index] += len(paper_list)

            for i in tqdm(range(len(paper_list))):
                # if i>1:
                #     break
                paper = paper_list[i]
                path = output_path + '/' + paper + '.json'
                if os.path.exists(path):
                    num_drafts[y,index]+=1
                else:
                    continue
                with open(path, 'r') as f:
                    data = json.load(f)
                if 'edits' not in data:
                    continue
                edits = data['edits']
                #compute paragraph level states
                texts = ["",""]
                for para_name in paras:
                    if para_name not in data:
                        continue
                    para = data[para_name]
                    if len(para)!=2:
                        continue
                    num_paras[0,index]+=1
                    content_before, content_after = para
                    texts[0]+= content_before
                    texts[0]+= " "
                    texts[1]+= content_after
                    texts[1]+= " "
                for k in range(2):
                    content = texts[k] 
                    sentences = tokenizer.tokenize(content)
                    filtered_sentences = filter_sentence(sentences, reg_set)
                    output = paragraph_summary_statistics(tool, filtered_sentences)
                    num_sen[k, index] += output['num_sen']
                    avg_sen_len[k, index] += output['avg_sen_len']
                    correctness_score[k, index] += output['correctness']
                    readability_score[k, index] += output['readability']

                    if k==0:
                        papers.append(paper)
                        read_scores.append(output['readability'])
                        correct_score.append(output['correctness'])
                        alltexts.append(" ".join(filtered_sentences))
                        results.append(index)

                        
                            
                flag = 0
                for sec_e in edits:
                    sec_e = clean_up_edits(sec_e)
                    for s in sec_e:
                        dis = edit_distance(s[0],s[1])
                        edit_dis[0,index]+=dis
                        flag = 1
                        num_pair[y,index]+=1
                        # bleu[0,index]+=bleu_score(s[0].split(),s[1].split())
                        for k in range(2):
                            edit_len[k,index]+= len(s[k].split())
                        delta[0,index]+=(len(s[1].split())-len(s[0].split()))
                num_selected[y,index]+=flag


    print("===================")
    print("total")
    print(total)
    print("num_drafts")
    print(num_drafts)
    print("num_selected")
    print(num_selected)
    print("num of pairs")
    print(num_pair)
    print("=====================")
    print("paragraph level stats")
    total_papers = np.sum(num_drafts,axis=0)
    print(num_sen[0,:]/total_papers)
    print(num_sen[1,:]/total_papers)
    print(avg_sen_len[0,:]/total_papers)
    print(avg_sen_len[1,:]/total_papers)
    print("correctness")
    print(correctness_score/total_papers)
    print("readability")
    print(readability_score/total_papers)


    print("===================")
    print("sentence level stats")
    print(edit_len[0,:]/np.sum(num_pair,axis=0))
    print(edit_len[1,:]/np.sum(num_pair,axis=0))
    print(delta/np.sum(num_pair,axis=0))
    print(edit_dis/np.sum(num_pair,axis=0))
    # print(clean_edits)
    write = True
    #write document features
    if write:
        with open(f'paper_features_2.tsv', 'w', encoding='UTF-8') as f:
            f.write('paperid\ttext\tcorrect\treadability\tacceptance')
            f.write("\n")
            for i in range(len(papers)):
                f.write("%s" % papers[i])
                f.write("\t")
                f.write(alltexts[i])
                f.write("\t")
                f.write("%s" % correct_score[i])
                f.write("\t")
                f.write("%s" %read_scores[i])
                f.write("\t")
                f.write("%s" %results[i])
                f.write("\n")
    # if write:
    #     with open(f'paper_extracted_features.tsv', 'w', encoding='UTF-8') as f:
    #         f.write('paperid\ttext\tcorrect\treadability\tacceptance')
    #         f.write("\n")
    #         for i in range(len(papers)):
    #             f.write("%s" % papers[i])
    #             f.write("\t")
    #             f.write("%s" %alltexts[i])
    #             f.write("\t")
    #             f.write("%s" % correct_score[i])
    #             f.write("\t")
    #             f.write("%s" %read_scores[i])
    #             f.write("\t")
    #             f.write("%s" %results[i])
    #             f.write("\n")


        # RANDOM_SEED=42
        # with open(f'edits_identify_dataset.tsv','w') as f:
        #     f.write('Sen 1\tSen 2\tLabel\tAcceptence')
        #     f.write("\n")
        #     for i in range(3):
        #         for item in clean_edits[i]:
        #             switch = random.randint(0, 1)
        #             if switch == 0:
        #                 f.write("%s" % item[0])
        #                 f.write("\t")
        #                 f.write("%s" % item[1])
        #                 f.write("\t")
        #                 f.write('0')
        #             else:
        #                 f.write("%s" % item[1])
        #                 f.write("\t")
        #                 f.write("%s" % item[0])
        #                 f.write("\t")
        #                 f.write('1')
        #             f.write("\t")
        #             f.write(str(i))
        #             f.write("\n")
        # f.close()
