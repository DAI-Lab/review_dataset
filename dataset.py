import argparse
import json
import os
from collections import defaultdict
from tqdm import tqdm

import pdftotext
import openreview
from diff_match_patch import diff_match_patch
from pdfminer.high_level import extract_text


dmp = diff_match_patch()


def get_reviews(paper_id):
    client = openreview.Client(baseurl='https://api.openreview.net', username=USER_NAME, password=PASSWORD)
    note = client.get_note(paper_id)
    paper_number = note.number
    forum_reviews = client.get_notes(
        invitation=f'ICLR.cc/2019/Conference/-/Paper{paper_number}/Official_Review')
    meta_reviews = client.get_notes(
        invitation=f'ICLR.cc/2019/Conference/-/Paper{paper_number}/Meta_Review')[0]

    reviews = []
    for n in forum_reviews:
        review_metadata = {
            'review': n.content['review'],
            'rating': n.content['rating'],
            'confidence': n.content['confidence']
        }
        reviews.append(review_metadata)
    
    decision = meta_reviews.content['recommendation']
    meta_review = {
        'review': meta_reviews.content['metareview'],
        'decision': decision,
        'confidence': meta_reviews.content['confidence']
    }
    forum_metadata = {
        'forum': paper_id,
        'reviews': reviews,
        'metareview': meta_review,
        'decision': decision,
    }
    return forum_metadata


def clean_text(text):
    tt = text.replace('-\n', '')
    tt = tt.replace('\n', ' ')
    return tt


def get_introduction(path):
    text = extract_text(path)
    text = clean_text(text)
    loc_begin = text.find("INTRODUCTION")
    for stopper in ('RELATED', 'PRELIMINAR'):
        loc_end = text.find(stopper)
        if loc_end != -1:
            break

    introduction = text[loc_begin:loc_end]
    return introduction


def get_full_sentence(text, loc, length):
    delimiter = '. '
    offset = len(delimiter)
    start = text.rfind(delimiter, 0, loc)
    end = text.find(delimiter, loc + length-1)
    return text[start+offset:end+offset]


def get_pairwise_sentence(paper_id):
    before = f'open_review/iclr_accept/data/{paper_id}_rev_0.pdf'
    after = f'open_review/iclr_accept/data/{paper_id}_rev_latest.pdf'
    text_before = get_introduction(before)
    text_after = get_introduction(after)   

    patches = dmp.patch_make(text_before, text_after)

    diffs = dmp.diff_main(text_before, text_after)
    dmp.diff_cleanupSemantic(diffs)
    output_diff_to_html(diffs, path='view.html')
    edits = set()
    for patch in patches:
        # output_diff_to_html(patch.diffs)
        sentence_before = get_full_sentence(text_before, patch.start1, patch.length1)
        sentence_after = get_full_sentence(text_after, patch.start2, patch.length2)
        # print('-: ' + sentence_before)
        # print('+: ' + sentence_after)
        # print('='*50)
        # print()
        edits.add((sentence_before, sentence_after))
    return list(edits)


def output_diff_to_html(diffs, path):
    html = dmp.diff_prettyHtml(diffs)
    with open(path, 'w') as f:
        f.write(html)


if __name__ == "__main__":
    path = 'open_review/iclr_accept/raw/paper_id.txt'
    paper_list = []
    with open(path, 'r') as f:
        paper_list = f.read().splitlines() 

    for paper_id in paper_list:
        edits = get_pairwise_sentence(paper_id)
        data = get_reviews(paper_id)
        data['data'] = edits
        with open(f'open_review/processed/{paper_id}.json', 'w') as f:
            json.dump(data,f)




