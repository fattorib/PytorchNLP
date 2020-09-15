#package imports
import numpy as np
import pandas as pd
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import DatasetBBC
import DistilBERTClassifier
import BERTClassifier


#Choices are BERT or DistilBERT
model_to_use = 'BERT'

#Loading test and train data
train_path = 'train.csv'
train_data = pd.read_csv(train_path)

train_labels = train_data['category']
train_inputs = train_data['text'].values.tolist()

dict_vals = {'tech':0, 'business':1, 'sport':2, 'entertainment':3, 'politics':4}
train_labels = train_labels.map(dict_vals).values.tolist()


test_path = 'test.csv'
test_data = pd.read_csv(test_path)

#Tokenizer wants lists not NumPy arrays
test_labels = test_data['category']
test_inputs = test_data['text'].values.tolist()

test_labels = test_labels.map(dict_vals).values.tolist()


if model_to_use == 'DistilBERT':
    #Distilbert tokenizer
    from transformers import DistilBertTokenizerFast
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    #Encoding training data and test data
    train_encodings = tokenizer(train_inputs,truncation=True, padding=True)
    test_encodings = tokenizer(test_inputs,truncation=True, padding=True)
    
else:
    #Bert tokenizer
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    #Encoding training data and test data
    train_encodings = tokenizer(train_inputs,truncation=True, padding=True)
    test_encodings = tokenizer(test_inputs,truncation=True, padding=True)

#Initializing dataset class
train_dataset = DatasetBBC.BBCNewsDataset(train_encodings,train_labels)
test_dataset = DatasetBBC.BBCNewsDataset(train_encodings,train_labels)

#Initializing Dataloader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

#Get device to train/test on
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#Start timing for training/testing process
t0 = time.time()

epochs = 2

#Training model
if model_to_use == 'DistilBERT':
    model = DistilBERTClassifier.DistilBertClassifier()
    #Detaching all BERT gradients
    for param in model.dstl.parameters():
        param.requires_grad = False
    
    #Pass model to current device
    model.to(device)
    model.train(epochs,train_loader,device)
    
else:
    model = BERTClassifier.BertClassifier()
    #Detaching all BERT gradients
    for param in model.bert.parameters():
        param.requires_grad = False
    
    #Pass model to current device
    model.to(device)
    model.train(epochs,train_loader,device)
    

#Testing model
from eval_funcs import eval_metrics

for param in model.parameters():
    param.requires_grad = False
    

eval_metrics(test_loader,device,model)
    
    
t1 = time.time()

total = round(t1-t0,2)
print()
print('The model you used was {}'.format(model_to_use))
print()
print('Training and testing model for {} epochs on a {}. It took {} seconds to complete this'.format(epochs, torch.cuda.get_device_name(),total))

    
    
    
    
    
    
    
    
    
    
    
    
    
    

