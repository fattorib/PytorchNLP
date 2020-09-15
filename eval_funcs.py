import numpy as np
from sklearn import metrics


def validate(dataloader,device,model):
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

def eval_metrics(dataloader,device,model):
    targets, outputs = validate(dataloader,device,model)
    outputs = np.argmax(np.array(outputs),axis = 1)
    
    accuracy = round(100*metrics.accuracy_score(targets, outputs),2)
    f1_score_micro = round(metrics.f1_score(targets, outputs, average='micro'),2)
    f1_score_macro = round(metrics.f1_score(targets, outputs, average='macro'),2)
    print()
    print('Performance metrics:')
    print(f"Accuracy: {accuracy}%")
    print(f"F1 Score (Micro): {f1_score_micro}")
    print(f"F1 Score (Macro): {f1_score_macro}")




