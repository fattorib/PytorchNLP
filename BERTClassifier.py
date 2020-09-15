import torch
import torch.nn as nn
import torch.nn.functional as F
import DatasetBBC
from transformers import BertModel, AdamW

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
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
    
    def train(self, epochs,dataloader,device):
        #Train model. Default lr works well
        optimizer = AdamW(self.parameters())  
        criterion = nn.CrossEntropyLoss()
        
        for e in range(0,epochs):
            running_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                #Pass all parametrs to GPU
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
        
                labels = batch['labels'].to(device)
                # print(torch.cuda.memory_allocated(device))
                
                outputs = self.forward(input_ids, attention_mask=attention_mask)
        
                loss = criterion(outputs,labels)
                
                loss.backward()
                
                running_loss += loss.item()
                
                optimizer.step()
            
            print('Training Loss:', running_loss/len(dataloader))
        