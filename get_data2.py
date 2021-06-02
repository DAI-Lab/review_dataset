import argparse
import json
import os
import re
import csv        

import math
from collections import defaultdict

from tqdm import tqdm
import numpy as np

# import pdftotext
# import openreview
# from diff_match_patch import diff_match_patch
from pdfminer.high_level import extract_text,extract_pages
from metrics import bleu_score
import requests

# from torchtext.data.metrics import bleu_score

import Levenshtein
import nltk
import spacy
from nltk import sent_tokenize, word_tokenize
# from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

from pytorch_pretrained_bert.tokenization import BertTokenizer
from bs4 import BeautifulSoup
# dmp = diff_match_patch()
USER_NAME = 'yis@mit.edu'
PASSWORD = 'Suny1713'

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    paragraph_open = re.compile('<p>')
    paragraph_end = re.compile('</p>')
    soup = BeautifulSoup(raw_html,'html.parser')
    to_extract = soup.findAll('formula')
    for item in to_extract:
        item.decompose()
    to_extract = soup.findAll('head')
    for item in to_extract:
        item.decompose()
    html = str(soup)
    html = re.sub(paragraph_open, ' ', html)
    html = re.sub(paragraph_end, ' ', html)
    cleantext = re.sub(cleanr, '', html)
    return cleantext

def xml_to_text(path):
    import xml.etree.ElementTree as ET
    import re
    sections = {}
    with open(path, "r") as inputFile: 
        lines = inputFile.readlines()
        for i in range(len(lines)):
            if len(sections) == 3:
                return sections
            line = lines[i]
            if '<abstract>' in line:
                sections['abstract'] = clean_text(cleanhtml(lines[i+1]))
            if '<body>' in line:
                sections['introduction'] = cleanhtml(lines[i+1])
            if line.startswith('<div'):
                if 'CONCLUSION' in line or 'DISCUSSION' in line:
                    sections['conclusion'] = clean_text(cleanhtml(line))
    return sections


                    
      



def clean_text(text):
    tt = text.replace('-\n', '')
    tt = tt.replace('\n', ' ')
    return tt


def build_tokenizer():
    extra_abbreviations = ['e.g','eg','i.e','vs','al','w.r.t','E.g','a.k.a','i.i.d','Sec','Fig','fig','c.f','viz','etc']
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentence_tokenizer._params.abbrev_types.update(extra_abbreviations)
    return sentence_tokenizer


def get_pairwise_edits(text_before, text_after, tokenizer):
    
    min_bleu = 0.5
    min_leven = 3

    sen_before = [sen.strip() for sen in tokenizer.tokenize(text_before)]
    sen_after = [sen.strip() for sen in tokenizer.tokenize(text_after)]
    # sen_before = [sent.string.strip() for sent in tokenizer(text_before).sents]
    # sen_after = [sent.string.strip() for sent in tokenizer(text_after).sents]

    
    w_size = 4
    edits = set()
    for i in range(len(sen_before)):
        start = max(0,i-w_size)
        end = min(i+w_size, len(sen_after))
        nei_bleus = []
        match_idx = []
        prev_sents_tok = word_tokenize(sen_before[i])
        # prev_sents_tok = sen_before[i].split()
        for j in range(start,end):
            post_sents_tok = word_tokenize(sen_after[j])
            # post_sents_tok = sen_after[j].split()
            bleu = bleu_score(prev_sents_tok,post_sents_tok)
            nei_bleus.append(bleu)
            match_idx.append(j)
        if not nei_bleus:
            continue
        max_bleu = max(nei_bleus)
        idx = nei_bleus.index(max_bleu)
        lev_dist = Levenshtein.distance(sen_before[i],sen_after[match_idx[idx]])
        if max_bleu>min_bleu and max_bleu<1.0 and lev_dist>min_leven:
            if i==0:
                context_before = 'NA'
            else:
                context_before = sen_before[i-1]
            if i==len(sen_before)-1:
                context_after = 'NA'
            else:
                context_after = sen_before[i+1]
            edits.add((sen_before[i],sen_after[match_idx[idx]],context_before,context_after))

    return list(edits)



def clean_up_edits(edits):
    filtered = []
    reg_set = "^[ A-Za-z0-9,.!%^&*()?/|:;_-]*$"
    stop_words = ['Figure ', 'Under review as', 'Published as', 'Table ','https']
    flag = False
    for e in edits:
        if len(e)<=3:
            continue
        if not e[0] or not e[1]:
            flag = True
        for s in stop_words:
            if s in e[0] or s in e[1]:
                flag = True
        # dist = compute_edit_distance(e[0], e[1])
        check_pre = re.match(reg_set, e[0])
        check_after = re.match(reg_set, e[1])
        if not check_pre or not check_after:
            flag = True
        #check if propoer tokenization:
        if not e[0][0].isupper() or not e[1][0].isupper():
            flag = True
        if not flag:
            filtered.append(e)
    return filtered

    


if __name__ == "__main__":
    years = [2018,2019,2020,2021]
    # years = [2018]
    tokenizer = build_tokenizer()
    # tokenizer = spacy.load('en_core_web_lg')
    alledits = []

    for year in years:
        print("==========================")
        print(year)
        if year == 2018:
            # sections = ['accepted-oral-papers']
            sections = ['accepted-poster-papers','accepted-oral-papers','rejected-papers','workshop-papers']
        elif year == 2019:
            # sections = ['accepted-poster-papers']
            sections = ['accepted-poster-papers','accepted-oral-papers','rejected-papers']
        elif year == 2020:
            sections = ['accept-spotlight','accept-talk','accept-poster','reject']
            # sections = ['accept-spotlight']
        elif year == 2021:
            #spotlight stuck at 74
            # sections = ['spotlight-presentations']
            sections = ['oral-presentations','spotlight-presentations','poster-presentations','withdrawn-rejected-submissions']

        prefix = f'open_review/ICLR{year}'
        
        for section in sections:
            if 'oral' in section or 'spotlight' in section or 'talk' in section:
                index = 0
            elif 'poster' in section or 'workshop' in section:
                index = 1
            else:
                index = 2
            sentences = []
            missing = []
            missing_single_para = []
            i = 0
            print(section)
            id_path = f'{prefix}/raw/{section}.txt'
            # id_path = f'{prefix}/{section}_errors_2.txt'
            data_path = f'open_review/new_process/ICLR{year}/data/{section}'
            output_path = f'open_review/new_processed/ICLR{year}/{section}'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            paper_list = []
            with open(id_path, 'r') as f:
                paper_list = list(set(f.read().splitlines()))
            print("total number of papers")
            print(len(paper_list))
            for paper_id in paper_list:
                # paper_id = 'B1QRgziT-'
                # if i >= 1:
                #     break
                print(i)
                print(paper_id)
                output_path_1 = f'open_review/processed/ICLR{year}/{section}'

                file_path = f'{output_path_1}/{paper_id}.json'
                if not os.path.exists(file_path):
                    try:
                        client = openreview.Client(baseurl='https://api.openreview.net', username=USER_NAME, password=PASSWORD)
                    except:
                        continue
                    note = client.get_note(paper_id)
                    paper_number = note.number
                    paper_title = note.content['title']
                    authors = note.content['authors']
                    data = {}
                    data['id'] = paper_id
                    data['title'] = paper_title
                    data['authors'] = authors
                    data['year'] = year
                    data['decision'] = section
                    # else:
                    #     with open(file_path, 'r+') as f:
                    #         data = json.load(f)
                    # data['reviews'] = get_reviews(paper_id, year = year)
                else:
                    with open(file_path) as f:
                        data = json.load(f)

                edits = []
                contexts = []
                pars = ['abstract','introduction','conclusion']
                only_edit = False

                before = f'{data_path}/{paper_id}_rev_0.tei.xml'
                after = f'{data_path}/{paper_id}_rev_latest.tei.xml'
                if not os.path.exists(before) or not os.path.exists(after):
                    print("missing files")
                    missing.append(paper_id) 
                    # continue
                else:
                    before_text = xml_to_text(before)
                    after_text = xml_to_text(after)
                    if len(before_text) == 0:
                        print("fail extraction from xml")
                        missing.append(paper_id) 
                        continue
                    for par in pars:
                        if par in before_text and par in after_text:
                            v0 = before_text[par]
                            v1 = after_text[par]
                            edit = get_pairwise_edits(v0, v1, tokenizer)
                            edits.append(edit)
                            data[par] = [v0,v1]
                        else:
                            if par not in data:
                                missing_single_para.append(paper_id) 
                                # print("not extract")
                            else:
                                v0 = data[par][0]
                                v1 = data[par][1]
                                edit = get_pairwise_edits(v0, v1, tokenizer)
                                edits.append(edit)
                    data['edits'] = edits

                with open(f'{output_path}/{paper_id}.json', 'w') as f:
                    json.dump(data,f)
                i+=1

                #write aggregate data
                for sec_e in edits:
                    sec_e = clean_up_edits(sec_e)
                    for e in sec_e:
                        row = [paper_id, e[0], e[1], e[2], e[3], index]
                        alledits.append(row)

            # write files with parsing issues
            with open(f'{prefix}/{section}_errors_2.txt', 'w') as f:
                for item in missing:
                    f.write("%s\n" % item)
            with open(f'{prefix}/{section}_missing_2.txt', 'w') as f:
                for item in missing_single_para:
                    f.write("%s\n" % item)
                # f.write('Missing single para')
                # for item in missing_single_para:
                #     f.write("%s\n" % item)


    with open(f'clean_edits_2.tsv', 'w',newline='') as f:
        f.write('Paper\tSen 1\tSen 2\tContext Before\tContext After\tAccept')
        f.write("\n")
        tsv_output = csv.writer(f, delimiter='\t')
        tsv_output.writerows(alledits)
    f.close()


