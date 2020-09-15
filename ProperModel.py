import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.nn.functional as F

text_path = 'train.csv'

data = pd.read_csv(text_path)


#Clearing GPU memory
torch.cuda.empty_cache()

#Transformer wants lists, not NumPy arrays...
train_labels = data['category']
train_inputs = data['text'].values.tolist()

dict_vals = {'tech':0, 'business':1, 'sport':2, 'entertainment':3, 'politics':4}

train_labels = train_labels.map(dict_vals).values.tolist()

from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

#Encoding training data and test data
train_encodings = tokenizer(train_inputs,truncation=True, padding=True)

#Creating torch dataset
class BBCNewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
    
#Initializing dataset class
train_dataset = BBCNewsDataset(train_encodings,train_labels)

#Initializing Dataloader
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


#Check if GPU available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#Model and optimizer inports
from transformers import DistilBertModel, AdamW

# model = DistilBertModel.from_pretrained('distilbert-base-uncased')


class DistiBertClassifier(nn.Module):
    
    def __init__(self):
        super(DistiBertClassifier, self).__init__()
        self.dstl = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.preclass = nn.Linear(768,768)
        self.classifier = nn.Linear(768,5)
        self.dropout = nn.Dropout(p =0.2)
                    
    def forward(self, input_ids, attention_mask):
        
        outputs = self.dstl(input_ids, attention_mask=attention_mask)
        outputs = outputs[0]
        outputs = outputs[:, 0]
        outputs = F.relu(self.preclass(outputs))
        outputs = self.dropout(outputs)
        outputs = self.classifier(outputs)
        
        return outputs

import time
t0 = time.time()
model=DistiBertClassifier()
# print(model.dstl)

for param in model.dstl.parameters():
    param.requires_grad = False
    
    
# model.dstl.requires_grad= False
model.to(device)        

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters())

epochs = 1

for e in range(0,epochs):
    running_loss = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        #Pass all parametrs to GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        labels = batch['labels'].to(device)
        # print(torch.cuda.memory_allocated(device))
        
        outputs = model(input_ids, attention_mask=attention_mask)

        loss = criterion(outputs,labels)
        
        loss.backward()
        
        running_loss += loss.item()
        
        optimizer.step()
    
    print('Training Loss:', running_loss/len(train_loader))
        





model.eval()

#Model Evaluation
test_path = 'test.csv'

data = pd.read_csv(test_path)


#Tokenizer wants lists not NumPy arrays
test_labels = data['category']
test_inputs = data['text'].values.tolist()

dict_vals = {'tech':0, 'business':1, 'sport':2, 'entertainment':3, 'politics':4}

test_labels = test_labels.map(dict_vals).values.tolist()

from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

#Encoding training data and test data
test_encodings = tokenizer(test_inputs,truncation=True, padding=True)

#Initializing dataset class
test_dataset = BBCNewsDataset(test_encodings,test_labels)

#Initializing Dataloader

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

#Remove all gradient tracking. Apparently this is more optimal?

for param in model.parameters():
    param.requires_grad = False



def validate(dataloader):
    targets_arr=[]
    outputs_arr=[]
    for batch in dataloader:
        #Pass all parametrs to GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
    
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        targets_arr.extend(labels.cpu().detach().numpy().tolist())
        outputs_arr.extend(outputs.cpu().detach().numpy().tolist())
        
        
    return targets_arr, outputs_arr



from sklearn import metrics
def eval_func():
    targets, outputs = validate(test_loader)
    outputs = np.argmax(np.array(outputs),axis = 1)
    
    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")
    
eval_func()   

t1 = time.time()

total = t1-t0
print('Training and testing model for {} epochs on {}. It took {} seconds to complete this'.format(epochs, torch.cuda.get_device_name(),total))

























