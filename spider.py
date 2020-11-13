
#!/usr/bin/python3

import os

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def get_paper_ids(section):
    
    url = f'https://openreview.net/group?id=ICLR.cc/2020/Conference#{section}'

    driver = webdriver.Firefox(executable_path='assets/geckodriver')

    driver.get(url)
    paper_ids = []
    try:
        html = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CLASS_NAME, "submissions-list"))
        )
        pdf_links = html.find_elements_by_class_name('pdf-link')
        for link in pdf_links:
            paper_id = link.get_attribute('href').split('id=')[-1].strip()
            paper_ids.append(paper_id)

    finally:
        driver.quit()
    
    write_path = f'open_review/iclr_accept/raw/{section}.txt'

    with open(write_path, 'a') as f:
        for paper_id in paper_ids:
            f.write(f"{paper_id}\n")
            




def get_revisions_from_paper_ids(section):
    with open(f'open_review/iclr_accept/raw/{section}.txt', 'r') as f:
        paper_ids = []
        for line in f.readlines():
            paper_ids.append(line.strip())


    paper_revisions = []

    driver = webdriver.Firefox(executable_path='assets/geckodriver')

    for paper_id in paper_ids[:20]:
        url = f'https://openreview.net/revisions?id={paper_id}'

        driver.get(url)
        try:
            html = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CLASS_NAME, "submissions-list"))
            )
            pdf_links = html.find_elements_by_class_name('attachment-download-link')
            lst = []
            for link in pdf_links:
                url = link.get_attribute('href')
                url = url.replace('&name=original_pdf', '').replace('attachment', 'pdf')
                lst.append(url)

            paper_revisions.append((paper_id, tuple(lst)))

        finally:
            pass

    driver.quit()

    folder = f'open_review/iclr_accept/raw/{section}'

    if not os.path.exists(folder):
        os.makedirs(folder)

    write_path = f'{folder}/revisions.txt'

    with open(write_path, 'a') as f:
        for paper_id, revisions in paper_revisions:
            f.write(f"\nid: {paper_id}\n")
            
            for idx, rev in enumerate(reversed(revisions)):
                f.write(f"rev {idx}: {rev}\n")



if __name__ == "__main__":
    # get_paper_ids(section = 'accept-spotlight')
    get_revisions_from_paper_ids(section = 'accept-spotlight')