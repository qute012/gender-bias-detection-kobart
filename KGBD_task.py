import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

from transformers import BartModel
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer

class KGBDDataset(Dataset):
    def __init__(self, data, max_seq_len=128):
        self.data = data
        self.max_seq_len = max_seq_len
        self.tokenizer = get_kobart_tokenizer()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        comments, label = data['comments'], data['contain_gender_bias']
        tokens = [self.tokenizer.bos_token] + \
            self.tokenizer.tokenize(comments) + [self.tokenizer.eos_token]
        encoder_input_id = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(encoder_input_id)
        if len(encoder_input_id) < self.max_seq_len:
            while len(encoder_input_id) < self.max_seq_len:
                encoder_input_id += [self.tokenizer.pad_token_id]
                attention_mask += [0]
        else:
            encoder_input_id = encoder_input_id[:self.max_seq_len - 1] + [
                self.tokenizer.eos_token_id]
            attention_mask = attention_mask[:self.max_seq_len]
        return {'input_ids': np.array(encoder_input_id, dtype=np.int_),
                'attention_mask': np.array(attention_mask, dtype=np.float),
                'labels': np.array(label, dtype=np.int_)}
                
import koco

train_dev = koco.load_dataset('korean-hate-speech', mode='train_dev')

batch_size = 16
train_dataset = KGBDDataset(train_dev['train'])
valid_dataset = KGBDDataset(train_dev['dev'])
train_dataloader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                num_workers=4, shuffle=True)
valid_dataloader = DataLoader(valid_dataset,
                                batch_size=batch_size,
                                num_workers=4, shuffle=False)

from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import BartForSequenceClassification

model = BartForSequenceClassification.from_pretrained(get_pytorch_kobart_model()).cuda()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(
        nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(
        nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters,
                  lr=5e-5, correct_bias=False)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(len(train_dataloader)*0.1), 
    num_training_steps=len(train_dataloader)
)

from tqdm import tqdm

best_acc = 0.0

for epoch in range(20):
    model.train()
    
    losses = 0.0
    iters = len(train_dataloader)
    pbar = tqdm(train_dataloader)
    for i,batch in enumerate(pbar):
        pbar.set_description('loss: {:.4f}'.format(losses/(i+1)))
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        
        optimizer.zero_grad()
        
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        labels = labels.to(model.device)
        outs = model(input_ids=input_ids, 
                     attention_mask=attention_mask, 
                     labels=labels, 
                     return_dict=True)
        loss = outs.loss
        
        loss.backward()
        optimizer.step()
        scheduler.step(epoch + i / iters)
        
        losses += loss.item()
        
    model.eval()
    
    acc = 0.0
    pbar = tqdm(valid_dataloader)
    for i,batch in enumerate(pbar):
        pbar.set_description('acc: {:.2f}'.format(acc/(i+1)))
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        labels = labels.to(model.device)
        outs = model(input_ids=input_ids, 
                     attention_mask=attention_mask, 
                     labels=None, 
                     return_dict=True)
        preds = torch.nn.functional.softmax(outs.logits, dim=1).max(dim=-1)[1]
        acc += (((preds==labels).sum())/preds.size(0)).item()
        
        losses += loss.item()
    
    if acc>best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'models/best_acc.pth')