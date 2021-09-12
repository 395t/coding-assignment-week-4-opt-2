#!/usr/bin/env python
# coding: utf-8

import sys
import os
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from tqdm.autonotebook import tqdm, trange #if this throws error use "from tqdm import tqdm"
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import AutoModel
from transformers import AutoTokenizer

import xclib.evaluation.xc_metrics as xc_metrics
import xclib.data.data_utils as data_utils
import pandas as pd
def printacc(score_mat, K = 5, X_Y = None, disp = True, inv_prop_ = -1):
    if X_Y is None: X_Y = tst_X_Y
    if inv_prop_ is -1 : inv_prop_ = inv_prop
        
    acc = xc_metrics.Metrics(X_Y.tocsr().astype(np.bool), inv_prop_)
    metrics = np.array(acc.eval(score_mat, K))*100
    df = pd.DataFrame(metrics)
    
    if inv_prop_ is None : df.index = ['P', 'nDCG']
    else : df.index = ['P', 'nDCG', 'PSP', 'PSnDCG']
        
    df.columns = [str(i+1) for i in range(K)]
    if disp: print(df.round(2))
    return df

trnX = [x.strip() for x in open('EUR-Lex/train_texts.txt').readlines()]
tstX = [x.strip() for x in open('EUR-Lex/test_texts.txt').readlines()]

trn_lbls = [x.strip().split() for x in open('EUR-Lex/train_labels.txt').readlines()]
tst_lbls = [x.strip().split() for x in open('EUR-Lex/test_labels.txt').readlines()]

vectorizer = CountVectorizer(lowercase=False, tokenizer=lambda x: x)
trn_X_Y = vectorizer.fit_transform(trn_lbls)
tst_X_Y = vectorizer.transform(tst_lbls)

Y = vectorizer.get_feature_names()
inv_prop = xc_metrics.compute_inv_propesity(trn_X_Y, 0.55, 1.5)
trn_X_Y, tst_X_Y

def tokenize(tokenizer, corpus, maxsize=512, bsz=1000):
    max_len = max(min(max([len(sen) for sen in corpus]), maxsize), 16)
    encoded_dict = {'input_ids': [], 'attention_mask': []}

    for ctr in tqdm(range(0, len(corpus), bsz)):
        temp = tokenizer.batch_encode_plus(
                corpus[ctr:min(ctr+bsz, len(corpus))],  # Sentence to encode.
                add_special_tokens = True,              # Add '[CLS]' and '[SEP]'
                max_length = max_len,                   # Pad & truncate all sentences.
                padding = 'max_length',
                return_attention_mask = True,           # Construct attn. masks.
                return_tensors = 'pt',                  # Return numpy tensors.
                truncation=True
        )
        encoded_dict['input_ids'].append(temp['input_ids'])
        encoded_dict['attention_mask'].append(temp['attention_mask'])

    encoded_dict['input_ids'] = torch.vstack(encoded_dict['input_ids'])
    encoded_dict['attention_mask'] = torch.vstack(encoded_dict['attention_mask'])
    return encoded_dict

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

class BertDataset(torch.utils.data.Dataset):
    def __init__(self, encoded_dict, X_Y):
        self.encoded_dict = encoded_dict
        self.X_Y = X_Y
        
    def __getitem__(self, index):
        return index
    
    def get_batch(self, indices):
        return { 'ii': self.encoded_dict['input_ids'][indices],
                 'am': self.encoded_dict['attention_mask'][indices],
                 'Y': torch.FloatTensor(self.X_Y[indices].toarray()),
                 'inds': indices}
    
    def __len__(self):
        return self.X_Y.shape[0]
    
class BertCollator():
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __call__(self, batch):
        return self.dataset.get_batch(batch)


class Net(nn.Module):
    def __init__(self, numy):
        super(Net, self).__init__()
        self.encoder = AutoModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(DROPOUT)
        self.w = nn.Linear(768, numy)
            
    def forward(self, x):
        embs = self.encoder(**x)['last_hidden_state'][:, 0]
        return self.w(self.dropout(embs))

def predict(net, tst_loader, tst_X_Y, K=5):
    data = np.zeros((tst_X_Y.shape[0], K))
    inds = np.zeros((tst_X_Y.shape[0], K)).astype(np.int32)
    indptr = np.arange(0, tst_X_Y.shape[0]*K+1, K)
    net.eval()
    
    with torch.no_grad():
        for b in tqdm(tst_loader, leave=True, desc='Evaluating'):
            out = net({'input_ids': b['ii'].to(DEVICE), 'attention_mask': b['am'].to(DEVICE)})
            top_data, top_inds = torch.topk(out, K)
            data[b['inds']] = top_data.detach().cpu().numpy()
            inds[b['inds']] = top_inds.detach().cpu().numpy()
            del top_data, top_inds, b, out
            
    torch.cuda.empty_cache() 
    score_mat = sp.csr_matrix((data.ravel(), inds.ravel(), indptr), tst_X_Y.shape)
    metrics = printacc(score_mat, X_Y=tst_X_Y, K=K);
    return score_mat, metrics

BSZ = 64
DROPOUT = 0.5
LR = 1e-4
NUM_EPOCHS = 25
EVAL_INTERVAL = 1
MAXLEN = 128
DEVICE='cuda:2'

LR_FACTORs = [10, 1, 0.1, 0.01, 0.001]
BETAs = [0.5, 0.75, 0.9, 0.99]
OPTIMs = [(torch.optim.Adagrad, {'lr': 1.0}), 
          (torch.optim.Adadelta, {'lr': 0.01}), 
          (torch.optim.Adam, {'lr': 0.001, 'amsgrad': False}),
          (torch.optim.Adam, {'lr': 0.001, 'amsgrad': True}),
          (torch.optim.AdamW, {'lr': 0.001})]

CRITERION = nn.BCEWithLogitsLoss(reduction='mean')
RES_DIR = 'Results'

trn_encoded_dict = tokenize(tokenizer, trnX, MAXLEN)
tst_encoded_dict = tokenize(tokenizer, tstX, MAXLEN)

trn_dataset = BertDataset(trn_encoded_dict, trn_X_Y)
trn_loader = torch.utils.data.DataLoader(
    trn_dataset,
    batch_size=BSZ,
    num_workers=1,
    collate_fn=BertCollator(trn_dataset),
    shuffle=True,
    pin_memory=True)

tst_dataset = BertDataset(tst_encoded_dict, sp.csr_matrix(tst_X_Y.shape))
tst_loader = torch.utils.data.DataLoader(
    tst_dataset,
    batch_size=BSZ,
    num_workers=1,
    collate_fn=BertCollator(tst_dataset),
    shuffle=False,
    pin_memory=True)

def train(net, optim):
    expname = ''.join(repr(optim).split('\n'))
    print(f'Training with {expname}')
    trn_metrics = [expname]
    for epoch in range(NUM_EPOCHS):
        net.train()
        cum_loss = 0; ctr = 0
        pbar = tqdm(trn_loader, leave=True)

        for b in pbar:
            optim.zero_grad()
            out = net({'input_ids': b['ii'].to(DEVICE), 'attention_mask': b['am'].to(DEVICE)})
            loss = CRITERION(out, b['Y'].to(DEVICE))
            loss.backward()
            optim.step()

            cum_loss += loss.item()
            ctr += 1
            pbar.set_description(f'Epoch: {epoch}/{NUM_EPOCHS}, Loss: {"%.4E"%(cum_loss/ctr)}', refresh=True)

        trn_metrics.append({'epoch': epoch, 'loss': cum_loss/ctr})
        if epoch%EVAL_INTERVAL == 0:
            score_mat, eval_metrics = predict(net, tst_loader, tst_X_Y)
            trn_metrics[-1]['tst P@1'] = eval_metrics['1']['P']
            trn_metrics[-1]['tst P@5'] = eval_metrics['5']['P']
            trn_metrics[-1]['tst PSP@1'] = eval_metrics['1']['PSP']
            trn_metrics[-1]['tst PSP@5'] = eval_metrics['5']['PSP']
            
    json.dump(trn_metrics, open(f'{RES_DIR}/{expname}.json', 'w'))
    return trn_metrics

all_metrics = {}
for optim_class, base_optim_params in OPTIMs:
    for lr_factor in LR_FACTORs:
        net = Net(trn_X_Y.shape[1]).to(DEVICE)
        params = base_optim_params.copy()
        params['lr'] *= lr_factor
        optim = optim_class(net.parameters(), **params)
        trn_metrics = train(net, optim)
        all_metrics[repr(optim)] = trn_metrics