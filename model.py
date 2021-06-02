import os
import re
import argparse
import csv
import torch
import random
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification, AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

from dataset import PaperDataset, EditDataset
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm
from metrics import compute_metrics
# from bert import data


class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name)

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]
        return loss, text_fea

def save_checkpoint(save_path, model, valid_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']

def evaluate(model, test_loader):
    y_pred = []
    y_true = []
    for batch in test_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels, return_dict=False)
        loss, output = outputs
        y_pred.extend(torch.argmax(output, 1).tolist())
        y_true.extend(labels.tolist())
    print('Classification Report:')
    metrics = compute_metrics(y_pred, y_true)
    print(metrics) 

def clean_txt(text):
    # text = re.sub("'","",text)
    text = re.sub("(\\W)+"," ",str(text))
    return text


if __name__=='__main__':
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--dataset', type=str, default='edit',
                        help='number of epochs')
    args = parser.parse_args()
    # tokenizer_name = 'bert-base-uncased'
    tokenizer_name = 'distilbert-base-uncased'
    tokenizer_path = f'tokenizer/{tokenizer_name}'
    
    if os.path.exists(tokenizer_path):
        # tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
    else:
        print("download tokenizer")
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenizer.save_pretrained(tokenizer_path)

    
    

    
    
    
    model_path = f'model/{tokenizer_name}'

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if os.path.exists(model_path):
        print("load pre-trained model")
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        # model = BertForSequenceClassification.from_pretrained(model_path,num_labels=num_label)
    else:
        print("download model")
        model = BertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        # model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=num_label)
        model.save_pretrained(model_path)

    if args.dataset == 'edit':
        batch_size = 32
        max_len = 128
        # df = pd.read_csv('edits_identify_dataset.tsv', sep='\t')
        df = pd.read_csv('clean_edits_2.tsv',sep='\t')
        #create swap index
        swap = np.random.randint(0,2,size=len(df))
        df['Label'] = swap
        # swap_index = np.where(swap==2)
        df.loc[df['Label']==1,['Sen 1','Sen 2']] = df.loc[df['Label']==1,['Sen 2','Sen 1']].values

        df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
        df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
        train_texts = list(zip(df_train['Sen 1'].tolist(), df_train['Sen 2'].tolist()))
        train_labels = df_train['Label'].tolist()
        val_texts = list(zip(df_val['Sen 1'].tolist(), df_val['Sen 2'].tolist()))
        val_labels = df_val['Label'].tolist()
        test_texts = list(zip(df_test['Sen 1'].tolist(), df_test['Sen 2'].tolist()))
        test_labels = df_test['Label'].tolist()
        train_dataset = EditDataset(train_texts, train_labels, tokenizer, max_len)
        val_dataset = EditDataset(val_texts, val_labels, tokenizer, max_len)
        test_dataset = EditDataset(test_texts, test_labels, tokenizer, max_len)
        optim = AdamW(model.parameters(), lr=3e-5)

    else:
        batch_size = 16
        max_len =500
        # data = pd.read_csv('paper_outcome.tsv', sep='\t')
        df = pd.read_csv('paper_features_2.tsv', sep='\t')
        df['label'] = df['acceptance']
        # data['text'] = data['abstract']
        # data['text'] = data['abstract']+data['introduction']+data['conclusion']
        # data['text'] = data['text'].apply(clean_txt)
        df = df[['text','label']]
        # df = df[df.text!='None']
        df = df[df.text!='NoneNoneNone']
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        labels = [int(x) for x in labels]
        num_label = 2
        labels = [1 if x==0 else x for x in labels]
        labels = [0 if x==2 else x for x in labels]
        train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=RANDOM_SEED)
        test_texts, val_texts, test_labels, val_labels = train_test_split(test_texts, test_labels, test_size=0.5, random_state=RANDOM_SEED)
  
        train_dataset = PaperDataset(train_texts, train_labels, tokenizer, max_len)
        val_dataset = PaperDataset(val_texts, val_labels, tokenizer, max_len)
        test_dataset = PaperDataset(test_texts, test_labels, tokenizer, max_len)
        optim = AdamW(model.parameters(), lr=3e-5)


    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)


    num_epochs = args.num_epochs
    running_loss = 0.0
    valid_running_loss = 0.0
    train_loss_list = []
    valid_loss_list = []

    for epoch in range(num_epochs):
        for batch in tqdm(train_loader):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()
            running_loss += loss.item()

        
        if epoch%1 == 0:
            model.eval()
            with torch.no_grad():                    
                # validation loop
                for val_batch in val_loader:
                    input_ids = val_batch['input_ids'].to(device)
                    attention_mask = val_batch['attention_mask'].to(device)
                    labels = val_batch['label'].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs[0]
                    valid_running_loss += loss.item()

            # evaluation
            average_train_loss = running_loss
            average_valid_loss = valid_running_loss
            train_loss_list.append(average_train_loss)
            valid_loss_list.append(average_valid_loss)

            # resetting running values
            running_loss = 0.0                
            valid_running_loss = 0.0
            best_valid_loss = float("Inf")
            model.train()

            # print progress
            print('Epoch [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                    .format(epoch+1, num_epochs, 
                            average_train_loss, average_valid_loss))
            
            # checkpoint
            if best_valid_loss > average_valid_loss:
                best_valid_loss = average_valid_loss
                save_checkpoint(f'log/model_{args.dataset}_{max_len}.pt', model, best_valid_loss)
    
    print('Finished Training!')
    evaluate(model, test_loader)