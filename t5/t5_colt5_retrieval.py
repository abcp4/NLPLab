import numpy as np
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
from torch.utils.data import DataLoader
import math 
from accelerate import Accelerator
import torch.nn.functional as F
# from einops import rearrange, pack, unpack
from feature_extractor import HiddenLayerExtractor
from x_transformers import TransformerWrapper, Decoder,Encoder
from process_tokenizer import get_tokenizer
from process_tokenizer import TokenizedCLMDataset,TokenizedMLMDataset

from utils import get_optimizer_scheduler

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
    VOCAB_SIZE = 16_000 # tokenmonster
    # VOCAB_SIZE = 1_024  # tokenmonster
    DEVICE_BATCH_SIZE = 1 # Token monster aguenta um batch size de (200-248)!! Geralmente melhora a qualidade do treino

    SEQ_LEN=84
    PRIMARY_SEQ_LEN=SEQ_LEN*5
    RETRIEVAL_NUM=10
    SECONDARY_SEQ_LEN=SEQ_LEN*RETRIEVAL_NUM
    MODEL_MAX_SEQ_LEN = PRIMARY_SEQ_LEN+SECONDARY_SEQ_LEN 
    
    # DEVICE_BATCH_SIZE=12
    # MODEL_MAX_SEQ_LEN = 1000

print('MODEL_MAX_SEQ_LEN: ',MODEL_MAX_SEQ_LEN)
N_DOMAINS=5


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
dataset,model_max_seq_len,norm,vocab=get_tokenizer(dataset,NUM_TOKENIZER_TRAINING_ITEMS,VOCAB_SIZE,TOKENIZER_PATH,'clm',PRIMARY_SEQ_LEN)
#dataset,model_max_seq_len,norm,vocab=get_tokenizer(dataset,NUM_TOKENIZER_TRAINING_ITEMS,VOCAB_SIZE,TOKENIZER_PATH,'clm',MODEL_MAX_SEQ_LEN+2)
tokenized_dataset = TokenizedCLMDataset(dataset,model_max_seq_len,norm,vocab)
#tokenized_dataset = TokenizedMLMDataset(dataset,model_max_seq_len,norm,vocab,MASK_ID,PAD_ID)

import string
from fastbm25 import fastbm25
import data_clusterize
is_domain=False

if is_domain:
    print('training vectorizer...')
    tfidf_model, vecs = data_clusterize.train_vectorizer(dataset['text'])
    print('clustering...')
    cluster_labels=data_clusterize.clusterize(vecs,N_DOMAINS=N_DOMAINS)
    print('clusterized!')
    domain_corpus={i:[] for i in range(N_DOMAINS)}
    
    for i in range(len(cluster_labels)):
        domain_corpus[cluster_labels[i]].append(dataset[i]['text'])
    print('Created domain corpus')
    domain_bm25=[]
    for i in range(N_DOMAINS):
        print('training bm25 for domain: ',i)
        corpus = domain_corpus[i]
        corpus = [doc.translate(str.maketrans('', '', string.punctuation)).replace('\n',"").lower().strip().split() for doc in corpus]
        domain_bm25.append(fastbm25(corpus))
    domain_corpus=[]
else:
    corpus = [text for text in dataset['text']]
    corpus = [doc.translate(str.maketrans('', '', string.punctuation)).replace('\n',"").lower().strip().split() for doc in corpus]
    bm25 = fastbm25(corpus)

colt5_dict={
            'encoder':{'is_colt5_encoder':True,'light_ff_mult':0.5,'heavy_ff_mult':4,'num_heavy_tokens':300,
                   'light_dim_head':64,'light_heads':8,'light_window_size':128,
                   'heavy_dim_head':64,'heavy_heads':8,'num_heavy_tokens_q':300,
                   'num_heavy_tokens_kv':300},
            'decoder':{'is_colt5_decoder':True,'light_ff_mult':0.5,'heavy_ff_mult':4,'num_heavy_tokens':300,
                    'light_dim_head':64,'light_heads':8,'light_window_size':128,
                    'heavy_window_size':128,'heavy_dim_head':64,'heavy_heads':8,
                    'num_heavy_tokens_q':32,'num_heavy_tokens_kv':300,'num_routed_kv':2, 
                    'use_triton':True, 'use_flash_attn':True},     
            }

encoder = TransformerWrapper(
    num_tokens = VOCAB_SIZE+8,
    max_seq_len = SEQ_LEN,
    attn_layers = Encoder(
        dim = int(768),
        depth = 3,
        heads =3,
        attn_flash = True,
        colt5_dict=colt5_dict['encoder'],
        # rel_pos_bias = True
    ),
    emb_dim=128,
)

decoder = TransformerWrapper(
    num_tokens = VOCAB_SIZE+8,
    max_seq_len = MODEL_MAX_SEQ_LEN,
    attn_layers = Decoder(
        dim = int(768),
        depth = 5,
        heads = 5,
        attn_flash = True,
        # rel_pos_bias = True
        colt5_dict=colt5_dict['decoder'],
        domains=N_DOMAINS,
    ),
    emb_dim=128,
    )

total_parameters=sum(p.numel() for p in decoder.parameters() if p.requires_grad)
print("Decoder params:", round(total_parameters/1e6,2),"M")
classifier = torch.nn.Linear(768, N_DOMAINS)
#vocab_output_layer = torch.nn.Linear(768, VOCAB_SIZE+8)
query_weights_layer = torch.nn.Linear(768, 1)
global_count=0
#given a query string, and a list of subwords, and embeddings for each subwords, return a list of embeddings for each word
def get_word2subword(query):
    tokens  = vocab.tokenize(query)
    tokens_str = [vocab.decode([int(i)]) for i in tokens]
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

    return word2subword_map 

#join subword embeddings
def get_words_embs(x,query):
    #query = vocab.decode([int(i) for i in query_ids[0]])

    query=' '.join(query.translate(str.maketrans('', '', string.punctuation)).replace('\n',"").lower().split())
    query=query.strip()
    words_embs=[]
    word2subword_map = get_word2subword(query)
    #print('x.shape: ',x.shape)
    #print('word2subword_map: ',word2subword_map)
    for i in range(len(word2subword_map)):
        word_map = word2subword_map[i]
        word = list(word_map.keys())[0]
        subword_ids = []
        for id in word_map[word]:
            #if id < MODEL_MAX_SEQ_LEN:
            if id < SEQ_LEN:
                subword_ids.append(id)
        subword_embs = x[:,subword_ids,:]
        word_emb = subword_embs.mean(dim=1)
        words_embs.append(word_emb)
    words_embs = torch.stack(words_embs,dim=1)
    #print('words_embs.shape: ',words_embs.shape)
    return words_embs

def search_similars(query,weights,domain=0,n=10):
    #print('query: ',query)
    tokenized_query =query.translate(str.maketrans('', '', string.punctuation)).replace('\n',"").lower().split()
    #print('tokenized_query: ',tokenized_query)
    #print(len(tokenized_query))
    if is_domain:
        result = domain_bm25[domain].top_k_sentence_weighted(tokenized_query,weights,k=n)
    else:
        result = bm25.top_k_sentence_weighted(tokenized_query,weights,k=n)
        # result = bm25.top_k_sentence(tokenized_query,k=n)

    sorted_scores=[r[1] for r in result]
    top_n = [dataset['text'][i] for i in sorted_scores[1:n+1]]
    return top_n
    
def trunc_pad(similar_docs,n=10):
    trunc_pad_similar_docs=[]
    c=0
    for doc in similar_docs:
        if c==0:
            doc='[RET]'+doc
        doc=vocab.tokenize(doc)[:SEQ_LEN]
        doc=list(doc)
        for j in range(SEQ_LEN-len(doc)):
            doc.append(PAD_ID)
        trunc_pad_similar_docs.append(doc)
        c+=1
    #caso não tenha docs suficientes
    for i in range(n-len(similar_docs)):
        trunc_pad_similar_docs.append([PAD_ID]*SEQ_LEN)

    tok = vocab.tokenize('[RET]')
    trunc_pad_similar_docs[-1][-len(tok):]=tok
    similar_docs=trunc_pad_similar_docs
    return similar_docs

class FusionInEncoder(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = HiddenLayerExtractor(encoder, layer=-2) 
        
        #feature extraction
        self.classifier = classifier
        self.vocab_output_layer = vocab_output_layer
        self.decoder = decoder

    def forward(self, x):
        #print('x.shape: ',x.shape)
        output_words= vocab.decode([int(i) for i in x[0]])
        similar_docs= search_similars(output_words,5)
        #trunc and pad
        similar_docs=trunc_pad(similar_docs)
        similar_docs = np.asarray(similar_docs,dtype=np.int32)
        similar_docs = torch.from_numpy(similar_docs).to(x.device)
        #print('similar_docs.shape: ',similar_docs.shape)

        #concatenate and flatten
        x = torch.cat((x,similar_docs),dim=0)
        x = x.flatten().unsqueeze(0)
        x = self.encoder(x)
        x = x[:,:MODEL_MAX_SEQ_LEN,:]
        output_logits=self.vocab_output_layer(x)
        # print('output_logits.shape: ',output_logits.shape)
        output_ids = output_logits.argmax(dim=2)
        
        #now get the domain
        x = x.mean(dim=1)
        x = self.classifier(x)
        x = F.softmax(x, dim=1)
        d=x.argmax(dim=1)

        x = self.decoder(output_ids,domain=d)
        return x

class FusionInDecoder(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = HiddenLayerExtractor(encoder, layer=-2) 
        #feature extraction
        self.classifier = classifier
        #self.vocab_output_layer = vocab_output_layer
        self.query_weights_layer = query_weights_layer
        self.decoder = decoder

    def forward(self, x):
        global global_count
        query_ids=x
        print('query_ids.shape: ',query_ids.shape)
        #x = self.encoder(x)
        x = self.encoder(x[:,:SEQ_LEN])

        #now get the domain
        x_m = x.mean(dim=1)
        x_m = self.classifier(x_m)
        x_m = F.softmax(x_m, dim=1)
        d=x_m.argmax(dim=1)
        #print('domain: ',d)

        #print('query_ids: ',query_ids)
        for i in range(len(query_ids[0])):
            if query_ids[0][i]>14990:
                query_ids[0][i]=0

        output_words= vocab.decode([int(i) for i in query_ids[0]])
        words_embs=get_words_embs(x,output_words)
        #print('words_embs.shape: ',words_embs.shape)
        query_weights=self.query_weights_layer(words_embs)[0]
        query_weights=query_weights.flatten()
        #convert with sigmoid
        query_weights=torch.sigmoid(query_weights)
        #query_weights=[1 for i in range(len(query_weights))]
        #benchmark time
        import timeit
        start = timeit.default_timer()
        similar_docs= search_similars(output_words,query_weights,d,RETRIEVAL_NUM)
        stop = timeit.default_timer()
        print('BM25 time: ', stop - start)
        global_count+=1
        if global_count%100==0:
            print('query: ',output_words)
            print('query_weights: ',query_weights)
            c=0
            for doc in similar_docs:
                c+=1
                print('doc',c,': ',doc)
        #trunc and pad
        similar_docs=trunc_pad(similar_docs,RETRIEVAL_NUM)
        similar_docs = np.asarray(similar_docs,dtype=np.int32)
        similar_docs = torch.from_numpy(similar_docs).to(x.device)

        #concatenate and flatten
        #no casual language modelling, nao faz sentido a query ficar na frente: o modelo nao ve os tokens futuros do bm25
        #A questao aqui tbm e que estamos fazendo lm na query e nas referencias. 
        #A outra opcao seria so a query no decoder,
        #fusionando os embeddings das referencias saidos do encoder.
        #Dessa forma, a tarefa seria ainda so prever a query, mas com embedings de referencias ajudando.
        similar_docs = similar_docs.reshape((1,similar_docs.shape[0]*similar_docs.shape[1]))
        query_ids = torch.cat((similar_docs[0],query_ids[0]),dim=0)
        #query_ids = torch.cat((similar_docs,query_ids),dim=0)
        query_ids = query_ids.flatten().unsqueeze(0)
        print('query_ids.shape: ',query_ids.shape)
        a=2/0
        
        #casual language modeling
        x = self.decoder(query_ids,domain=d)
        labels =torch.cat((query_ids[:,1:],torch.zeros((query_ids.shape[0],1),dtype=torch.long).to(x.device)),dim=1) 
        
        #FiD com Encoder no final: tarefa tem que ser mlm nesse caso. Vantagem e que a tarefa
        #e mais narutal que casual lm aplicada a um monte de passagens, o que leva o modelo a
        # aprender descontinuidades nos textos


        #masked language modeling
        #x = self.encoder2(query_ids,domain=d)

        return x,labels

#model = FusionInEncoder(encoder, decoder)
model = FusionInDecoder(encoder, decoder)
# input_ids are the indices corresponding to each token in the sentence.
# attention_mask indicates whether a token should be attended to or not.
# token_type_ids identifies which sequence a token belongs to when there is more than one sequence

train_dataloader = DataLoader(
        tokenized_dataset, shuffle=True, batch_size=DEVICE_BATCH_SIZE
    )
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
print("Now training!!")
for epoch in range(starting_epoch, num_train_epochs):
    model.train()
    for step, batch in enumerate(train_dataloader):
        labels=batch['labels'] 
        print('input_ids.shape: ',batch['input_ids'].shape)
        # with accelerator.accumulate(model):
        logits,labels=model(batch['input_ids']) 
        loss = F.cross_entropy(logits.transpose(1, 2),labels,ignore_index = PAD_ID)
        count_amostra+=int(len(labels))
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

