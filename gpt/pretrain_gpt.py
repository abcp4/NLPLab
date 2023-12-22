import torch
from torch import nn
from reformer_pytorch import ReformerLM
from x_transformers import TransformerWrapper, Decoder,Encoder
import numpy as np
from accelerate import Accelerator
import math 
from transformers import (
    get_scheduler,
)
import torch.nn.functional as F
from einops import rearrange, pack, unpack
from torch.utils.data import DataLoader
import os

DEVICE_BATCH_SIZE = 10
MAX_SEQ_LEN =60 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model = TransformerWrapper(
    num_tokens = 120,
    max_seq_len = MAX_SEQ_LEN,
    attn_layers = Decoder(
        dim = 256,
        depth = 12,
        heads = 4,
        attn_flash = True,
        # rel_pos_bias = True 
    )
)
tokenized_dataset=torch.tensor(np.load('/home/kiki/dados_servidor/Servidor/vida/dataset_hrs.npy'))
#tokenized_dataset = torch.tensor(np.random.randint(0, 120, (100, MAX_SEQ_LEN)), dtype=torch.long)

print(tokenized_dataset.shape)

#custom dataset class for casual language modeling(half sequence is used as input and other half as target)
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset):
        self.tokenized_dataset = tokenized_dataset

    def __getitem__(self, idx):
        return self.tokenized_dataset[idx, :MAX_SEQ_LEN//2], self.tokenized_dataset[idx, MAX_SEQ_LEN//2:]

    def __len__(self):
        return len(self.tokenized_dataset)

dataset = CustomDataset(tokenized_dataset)
train_dataloader = DataLoader(
        dataset, shuffle=True, batch_size=DEVICE_BATCH_SIZE
    )
# Optimizer
learning_rate=5e-5
weight_decay=0

gradient_accumulation_steps=1
max_train_steps=None
num_train_epochs=1
lr_scheduler_type='linear'
num_warmup_steps=0
# Split weights in two groups, one with weight decay and the other not.
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

# Scheduler and math around the number of training steps.
overrode_max_train_steps = False
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
if max_train_steps is None:
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True

lr_scheduler = get_scheduler(
    name=lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps * gradient_accumulation_steps,
    num_training_steps=max_train_steps * gradient_accumulation_steps,
)

accelerator = Accelerator()
# Prepare everything with our `accelerator`.
model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)

# Only show the progress bar once on each machine.
# progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
completed_steps = 0
starting_epoch = 0

# update the progress_bar if load from checkpoint
total_loss=0
# progress_bar.update(completed_steps)
steps_log=100
count_amostra=0
num_train_epochs=1
for epoch in range(starting_epoch, num_train_epochs):
    model.train()
    for step, batch in enumerate(train_dataloader):
        x, y = batch
        # with accelerator.accumulate(model):
        results=model(x) 
        c_loss = F.cross_entropy(
            results.transpose(1, 2),
            y,
        )
        count_amostra+=int(len(batch))
        loss = c_loss
        total_loss += loss.detach().float().cpu().numpy().item()
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()       
        
        if step%steps_log==0:
            print('iteration: ',step,', total_loss: ',total_loss/steps_log)
            print('count_amostra:',count_amostra)
            total_loss=0
        break

#now save model
torch.save(model.state_dict(), 'modelo_transformer.pth')
