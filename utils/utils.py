import math
from transformers import (
    get_scheduler,
)
import torch
def get_optimizer_scheduler(model,train_dataloader,gradient_accumulation_steps,learning_rate=5e-5,weight_decay=0, num_warmup_steps=0, max_train_steps=None,lr_scheduler_type='linear',num_train_epochs=1):
    # Optimizer
    
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
    return optimizer,lr_scheduler,max_train_steps


#given a query string, and a list of subwords, and embeddings for each subwords, return a list of embeddings for each word
def get_word2subword(vocab,query):
    tokens  = vocab.tokenize(query)
    # tokens_str = [vocab.decode([int(i)]) for i in tokens]
    #for i in range(len(tokens_str)):
    #    print('token',i,':',tokens_str[i])
    sentence_token_id = 0
    decoder = vocab.decoder()
    word2subword_map = [{p:[]} for p in query.split()]

    temp_word = ""
    temp_subwords = []

    for id, token_id in enumerate(tokens):
      token = decoder.decode(tokens[id]).strip()
      subwords = token.split()

      for sub_id, sub in enumerate(subwords):
        word_map = word2subword_map[sentence_token_id]

        if sub in word_map:
          word_map[sub] = [id]
          sentence_token_id += 1
        else:
          temp_word += sub
          temp_subwords.append(id)

        if temp_word in word_map:
            word_map[temp_word] = temp_subwords
            temp_word = ""
            temp_subwords = []
            sentence_token_id += 1

    return tokens.tolist(),word2subword_map 