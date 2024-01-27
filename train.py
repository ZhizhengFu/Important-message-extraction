import torch
import re
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import json
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_data(data):

    inputs = tokenizer.batch_encode_plus([re.sub(r'\[.*?\]|@全体成员', '', sample['text']) for sample in data],\
                                        add_special_tokens=True,\
                                        max_length=512,\
                                        return_tensors='pt',\
                                        padding='max_length',\
                                        truncation=True)

    return inputs

tokenizer = BertTokenizer.from_pretrained(\
        pretrained_model_name_or_path='/home/charles/code/bert-base-cased')

model = BertForSequenceClassification.from_pretrained(\
        pretrained_model_name_or_path='/home/charles/code/bert-base-cased',\
        num_labels=2)

model.to(device)

f = open('/home/charles/code/my_test.json','r')

content = f.read()

test_data = json.loads(content)

f = open('/home/charles/code/my_json.json','r')

content = f.read()

train_data = json.loads(content)

train_inputs = preprocess_data(train_data)

train_labels = torch.tensor([sample['label'] for sample in train_data])

train_dataset = torch.utils.data.TensorDataset(train_inputs['input_ids'],\
                                               train_inputs['attention_mask'],\
                                               train_labels)

train_dataloader = DataLoader(train_dataset,\
                              batch_size=8,\
                              shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

def train(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    total_samples = 0

    for batch in dataloader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch 
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        total_loss += loss.item()
        total_samples += len(input_ids)

        loss.backward()
        optimizer.step()

    return total_loss, total_samples

num_epochs = 10

def predict(model,tdata):

    test_lab = torch.tensor([sample['label'] for sample in tdata])

    model.eval()

    with torch.no_grad():
        inputs = preprocess_data(tdata)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits 
        
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    preds = np.argmax(probs, axis=1)

    return f1_score(test_lab, preds)

for epoch in range(num_epochs):
    total_loss, total_samples = train(model, train_dataloader, optimizer)
    tloss = (total_loss / total_samples) * 100
    path = '/home/frazier/code/' + 'outputdr' + str(epoch)
    print(f"Epoch {epoch+1}: Loss = {tloss}% accu = {predict(model,test_data)}")
    torch.save(model.state_dict(),path)
