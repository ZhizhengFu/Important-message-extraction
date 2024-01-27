import torch
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='/home/charles/code/bert-base-cased')
device = torch.device('cuda:0')
from datasets import load_from_disk
datasets = load_from_disk('/home/charles/code/data/glue_sst2')
def f(data):
    return tokenizer(
        data['sentence'],
        padding='max_length',
        truncation=True,
        max_length=50,
    )

datasets = datasets.map(f, batched=True, batch_size=1000, num_proc=4)
dataset_train = datasets['train'].shuffle()
print(dataset_train)
dataset_test = datasets['validation'].shuffle()
del datasets

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path='/home/charles/code/bert-base-cased',
                                                           num_labels=2)
model.to(device)
import numpy as np
from evaluate import load
from transformers.trainer_utils import EvalPrediction
metric = load(path='/home/charles/code/accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = logits.argmax(axis=1)
    return metric.compute(predictions=logits, references=labels)

eval_pred = EvalPrediction(
    predictions=np.array([[0, 1], [2, 3], [4, 5], [6, 7]]),
    label_ids=np.array([1, 1, 1, 1]),
)

print(compute_metrics(eval_pred))

from transformers import TrainingArguments, Trainer

args = TrainingArguments(output_dir='./output_dir',
                         evaluation_strategy='epoch',
                         per_device_eval_batch_size=32,
                         per_device_train_batch_size=16,
                         weight_decay=1e-2,
                         learning_rate=1e-4,
                         num_train_epochs=2)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    compute_metrics=compute_metrics,
)

print(trainer.evaluate())
trainer.train()
trainer.evaluate()
torch.save(model.state_dict(),'outputdr')

def collate_fn(data):
    label = [i['label'] for i in data]
    input_ids = [i['input_ids'] for i in data]
    token_type_ids = [i['token_type_ids'] for i in data]
    attention_mask = [i['attention_mask'] for i in data]

    label = torch.LongTensor(label).to(device)
    input_ids = torch.LongTensor(input_ids).to(device)
    token_type_ids = torch.LongTensor(token_type_ids).to(device)
    attention_mask = torch.LongTensor(attention_mask).to(device)

    return label, input_ids, token_type_ids, attention_mask

loader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                                          batch_size=4,
                                          collate_fn=collate_fn,
                                          shuffle=True,
                                          drop_last=True)

for i, (label, input_ids, token_type_ids, attention_mask) in enumerate(loader_test):
    break

def test():
    #加载参数
    model.load_state_dict(torch.load('outputdr'))

    model.eval()

    #运算
    out = model(input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask)

    #[4, 2] -> [4]
    out = out['logits'].argmax(dim=1)

    correct = (out == label).sum().item()

    return correct / len(label)

print(test())
