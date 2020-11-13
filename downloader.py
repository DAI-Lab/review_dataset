#!/usr/bin/python3

import os

import requests



def download_pdf_to_path(url, path):
    print(f"Downloading {url} to {path} ...")
    r = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        f.write(r.content)

def download_revisions(section):
    REVISION_FILE_PATH = f'open_review/iclr_accept/raw/{section}/revisions.txt'

    DOWNLOAD_MODE = 'FIRST_AND_LAST'  #  FIRST_AND_LAST or ALL

    paper_revisions = []

    with open(REVISION_FILE_PATH, 'r') as f:
        paper_id = ''
        for line in f.readlines():
            txt = line.strip()
            if txt.startswith('id:'):
                paper_id = txt.split('id: ')[1].strip()
                paper_revisions.append((paper_id, []))
            elif txt.startswith('rev'):
                pdf_link = txt.split(': ')[1].strip()
                paper_revisions[-1][1].append(pdf_link)



    folder = f'open_review/iclr/data/{section}'

    if not os.path.exists(folder):
        os.makedirs(folder)

    for paper_id, revs in paper_revisions:
        if DOWNLOAD_MODE == 'FIRST_AND_LAST':
            to_downloads = [('rev_0', revs[0]), ('rev_latest', revs[-1])]
        elif DOWNLOAD_MODE == 'ALL':
            to_downloads = [(f'rev_{i}', rev) for i, rev in enumerate(revs)]

        for rev_id, url in to_downloads:
            file_name = f'{paper_id}_{rev_id}'
            path = f'{folder}/{file_name}.pdf'
            download_pdf_to_path(url, path)



if __name__ == "__main__":
    download_revisions(section = 'accept-spotlight')