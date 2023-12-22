import torch
print(torch.__version__)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.is_available()

from pathlib import Path
import time

import datasets
from tokenizers import BertWordPieceTokenizer, Regex, normalizers
import torch.nn.functional as F
# from reformer_pytorch import ReformerLM
# from x_transformers import TransformerWrapper, Decoder,Encoder
from torch.utils.data import DataLoader
import math 

from accelerate import Accelerator
import torch.nn.functional as F
# from einops import rearrange, pack, unpack

model_training='tokemonster'#'bert' , 'tokemonster
# LIMIT_DATASET = 2016 * 4  # keep small for development, set to None for full dataset
#(3090) Para 128 tokens(Bert) e 32000 vocab: 1000 = 6 seg, 10000 = 1min, 100000 = 10min, 1m = 100min, 10m = 16h, 100m = 160h ou 6.6 dias
#(3090) Para 84 tokens(tokenmonster salva 35%) e 24000 vocab(Bert): 1000 = 4.3s, 10000 = 43s, 100000 = 7.1m, 1m = 71min, 10m = 11.8h, 100m = 118h ou 5 dias
#(4090) Para 84 tokens(tokenmonster salva 35%) e 24000 vocab(Electra): 1000 = 3.5s, 10000 = 35s, 100000 = 5.9m, 1m = 59min, 10m = 9.8h, 100m = 98h ou 4 dias
#(4090) Para 84 tokens(tokenmonster salva 35%) e 16000 vocab (Electra): 1000 = 2.1s, 10000 = 21s, 100000 = 3.53m, 1m = 35.3min, 10m = 5.8h, 100m = 58h ou 2.4 dias

#com retnet. Confirmado que é O(n). Se lembrando que não existe mta vantagem além do custo crescer sequencialmente, já que o conteudo 
#(4090) Para 84 tokens(tokenmonster salva 35%) batch size 200 e 16000 vocab (Electra): 1000 = 2.1s, 10000 = 21s, 100000 = 3.53m, 1m = 35.3min, 10m = 5.8h, 100m = 58h ou 2.4 dias
#(4090) Para 168 tokens(tokenmonster salva 35%) batch size 100 e 16000 vocab (Electra): 1000 = 3.7s, 10000 = 37.5s,
#(4090) Para 1000 tokens(tokenmonster salva 35%) batch size 12 e 16000 vocab (Electra):            , 10000 = 4min32s,

LIMIT_DATASET = 10_000
RANDOM_SEED = 42
NUM_TOKENIZER_TRAINING_ITEMS = 1_000_000  # I made this up, but it seems reasonable
if model_training=='bert':
    VOCAB_SIZE = 32_768  # from Cramming
    DEVICE_BATCH_SIZE = 100 # aprox 128, adjust to get near 100% gpu memory use
    MODEL_MAX_SEQ_LEN = 128  # from Cramming
else:
    VOCAB_SIZE = 16_000  # tokenmonster
    # VOCAB_SIZE = 1_024  # tokenmonster
    DEVICE_BATCH_SIZE = 100 # Token monster aguenta um batch size de (200-248)!! Geralmente melhora a qualidade do treino
    MODEL_MAX_SEQ_LEN = 84  # token_monster

    # DEVICE_BATCH_SIZE=12
    # MODEL_MAX_SEQ_LEN = 1000

## write a function that sum the    

MASK_ID=4
PAD_ID=0
gradient_accumulation_steps = 2048 // DEVICE_BATCH_SIZE  # roughly based on Cramming
batch_size = DEVICE_BATCH_SIZE * gradient_accumulation_steps
print('batch_size: ',batch_size)
RUN_DIR = Path("data") / f"run_{time.strftime('%Y%m%d-%H%M%S')}"
CHECKPOINT_DIR = RUN_DIR / "training_checkpoints"
MODEL_DIR = RUN_DIR / "model"
TOKENIZER_PATH = RUN_DIR / "tokenizer.json"
TRAINER_HISTORY_PATH = RUN_DIR / "trainer_history.json"

RUN_DIR.mkdir(exist_ok=True, parents=True)

dataset = datasets.load_dataset(
    "sradc/chunked-shuffled-wikipedia20220301en-bookcorpusopen",
    split=f"train[:{LIMIT_DATASET}]" if LIMIT_DATASET else "train",
    revision="0e6fada2dd43136e4a3f637da41de2e596aee674",
)
print('loaded dataset!!')

from process_tokenizer import get_tokenizer
tokenizer,tokenized_dataset,norm,vocab=get_tokenizer(dataset,NUM_TOKENIZER_TRAINING_ITEMS,VOCAB_SIZE,TOKENIZER_PATH,'clm',MODEL_MAX_SEQ_LEN+2)


from x_transformers import TransformerWrapper, Decoder,Encoder

model = TransformerWrapper(
    num_tokens = VOCAB_SIZE+8,
    max_seq_len = MODEL_MAX_SEQ_LEN,
    attn_layers = Decoder(
        dim = int(768),
        depth = 12,
        heads = 12,
        attn_flash = True,
        # rel_pos_bias = True
        is_colt5=True, 
    ),
    emb_dim=128,
)


# input_ids are the indices corresponding to each token in the sentence.
# attention_mask indicates whether a token should be attended to or not.
# token_type_ids identifies which sequence a token belongs to when there is more than one sequence

train_dataloader = DataLoader(
        tokenized_dataset, shuffle=True, batch_size=DEVICE_BATCH_SIZE
    )
from utils import get_optimizer_scheduler
optimizer,lr_scheduler,max_train_steps = get_optimizer_scheduler(model,train_dataloader,gradient_accumulation_steps,learning_rate=5e-5,weight_decay=0, num_warmup_steps=0, max_train_steps=None,lr_scheduler_type='linear',num_train_epochs=1)

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
steps_log=30
count_amostra=0
num_train_epochs=1
for epoch in range(starting_epoch, num_train_epochs):
    model.train()
    for step, batch in enumerate(train_dataloader):
        # with accelerator.accumulate(model):
        logits=model(batch['input_ids']) 
        loss = F.cross_entropy(logits.transpose(1, 2),batch['input_ids'],ignore_index = PAD_ID)
        count_amostra+=int(len(batch['input_ids']))
        # print(loss)
        total_loss += loss.detach().float().cpu().numpy().item()
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()       
        
        if step%steps_log==0:
            print('iteration: ',step,', total_loss: ',total_loss/steps_log)
            print('count_amostra:',count_amostra)
            total_loss=0
    # Checks if the accelerator has performed an optimization step behind the scenes
    if accelerator.sync_gradients:
        # progress_bar.update(1)
        completed_steps += 1

    if completed_steps >= max_train_steps:
        break

    model.eval()
    losses = []
