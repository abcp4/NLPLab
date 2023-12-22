
from tokenizers import BertWordPieceTokenizer, Regex, normalizers
from magic_timer import MagicTimer
from typing import Iterator
from tqdm import tqdm

from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import torch
import numpy as np


def tokenizer_training_data(dataset,num_tokenizer_train_items,n_dataset) -> Iterator[str]:
    for i in tqdm(
        range(min(num_tokenizer_train_items, n_dataset)),
        desc="Feeding samples to tokenizer",
    ):
        yield dataset[i]["text"]

class HFTokenizedDataset(torch.utils.data.Dataset):
    "This wraps the dataset and tokenizes it, ready for the model"

    def __init__(self, dataset, tokenizer,model_max_seq_len):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.model_max_seq_len=model_max_seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        inps= self.tokenizer.encode(
            self.dataset[i]["text"],
            return_tensors="pt",
            truncation=True,
            max_length=self.model_max_seq_len - 2,
            padding="max_length",
            return_special_tokens_mask=True,
        )[0, ...]
        return {'input_ids':inps}

#span masked language modelling
class TokenizedSMLMDataset(torch.utils.data.Dataset):
    def __init__(self, dataset,model_max_seq_len,norm,vocab,tokenizer,mask_id,pad_id,tok_type):
        self.dataset = dataset
        self.model_max_seq_len=model_max_seq_len
        self.norm=norm
        self.vocab=vocab
        self.tokenizer=tokenizer
        self.mask_id=mask_id
        self.pad_id=pad_id
        self.tok_type=tok_type

    def __len__(self):
        return len(self.dataset)
    
    def convert_tokens_to_ids(self,s):
        s=self.norm.normalize_str(s)
        tokens = self.vocab.tokenize(s).tolist()
        tokens=tokens[:self.model_max_seq_len - 2]
        tokens=torch.Tensor(tokens).long()
        return tokens
    
    def __getitem__(self, i):
        
        if self.tok_type=='hf':
            #hugginface tokenizer
            tokens=self.tokenizer(self.dataset[i]["text"])['input_ids']
            # print(tokens)
        elif self.tok_type=='tokenmonster':
            s=self.norm.normalize_str(self.dataset[i]["text"])
            tokens = self.vocab.tokenize(s).tolist()
        
        #trucate
        tokens=tokens[:self.model_max_seq_len - 2]
        l=len(tokens)
        #padding
        for j in range(l,self.model_max_seq_len - 2):
            tokens.append(0)
        tokens=torch.as_tensor(tokens,dtype=torch.long)

        #span masked language modelling
        labels=tokens.clone()
        unmasked_tokens=tokens.clone()
        total_budget = int(0.15 * len(tokens))
        current_budget = 0
        spans_start=''
        spans_end=''
        spans_tokens=''
        while current_budget < total_budget:
            span_length = np.random.geometric(p=0.2)
            if span_length > 10:
                span_length = 10
            
            if span_length > total_budget - current_budget:
                span_length = total_budget - current_budget
            span_start = np.random.randint(0, len(tokens) - span_length)
            labels[span_start : span_start + span_length] = -100
            #80% of the time, replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & (labels == -100)
            tokens[indices_replaced] = self.mask_id
            #10% of the time, keep original ([MASK]) tokens
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & (labels == -100) & ~indices_replaced
            if self.tok_type=='tokenmonster':
                random_words = torch.randint(len(self.vocab), labels.shape, dtype=torch.long)
            elif self.tok_type=='hf':
                random_words = torch.randint(0, len(self.tokenizer), labels.shape, dtype=torch.long)
            tokens[indices_random] = random_words[indices_random]
            
            current_budget += span_length
            # spans_start.append(span_start)
            # spans_end.append(span_start+span_length)
            spans_start+=str(span_start)+' '
            spans_end+=str(span_start+span_length)+' '
        
        return {'input_ids':tokens,
                'labels':labels,
                'unmasked_tokens':unmasked_tokens,
                'index':i,
                'spans_start':spans_start,
                'spans_end':spans_end}


#masked language modelling
class TokenizedMLMDataset(torch.utils.data.Dataset):
    def __init__(self, dataset,model_max_seq_len,norm,vocab,tokenizer,mask_id,pad_id,tok_type):
        self.dataset = dataset
        self.model_max_seq_len=model_max_seq_len
        self.norm=norm
        self.vocab=vocab
        self.mask_id=mask_id
        self.pad_id=pad_id
        self.tok_type=tok_type
        self.tokenizer=tokenizer

    def __len__(self):
        return len(self.dataset)

    def convert_tokens_to_ids(self,s):
        s=self.norm.normalize_str(s)
        tokens = self.vocab.tokenize(s).tolist()
        tokens=tokens[:self.model_max_seq_len - 2]
        tokens=torch.Tensor(tokens).long()
        return tokens

    def __getitem__(self, i):
        if self.tok_type=='hf':
            #hugginface tokenizer
            tokens=self.tokenizer(self.dataset[i]["text"])['input_ids']
            # print(tokens)
        elif self.tok_type=='tokenmonster':
            s=self.norm.normalize_str(self.dataset[i]["text"])
            tokens = self.vocab.tokenize(s).tolist()
        #trucate
        tokens=tokens[:self.model_max_seq_len - 2]
        l=len(tokens)
        #padding
        for j in range(l,self.model_max_seq_len - 2):
            tokens.append(0)
        tokens=torch.as_tensor(tokens,dtype=torch.long)

        #mlm
        labels=tokens.clone()
        probability_matrix = torch.full(labels.shape, 0.15)
        probability_matrix.masked_fill_(labels == self.pad_id, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        #80% of the time, replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        tokens[indices_replaced] = self.mask_id

        #10% of the time, keep original ([MASK]) tokens
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        randow_words = torch.randint(len(self.vocab), labels.shape, dtype=torch.long)
        tokens[indices_random] = randow_words[indices_random]

        return {'input_ids':tokens,
                'labels':labels,
                'index':i}

#casual language modelling
class TokenizedCLMDataset(torch.utils.data.Dataset):
    def __init__(self, dataset,model_max_seq_len,norm,vocab,tokenizer,mask_id,pad_id,tok_type):
        self.dataset = dataset
        self.model_max_seq_len=model_max_seq_len
        self.norm=norm
        self.vocab=vocab
        self.mask_id=mask_id
        self.pad_id=pad_id
        self.tok_type=tok_type
        self.tokenizer=tokenizer


    def __len__(self):
        return len(self.dataset)

    def convert_tokens_to_ids(self,s):
        s=self.norm.normalize_str(s)
        tokens = self.vocab.tokenize(s).tolist()
        tokens=tokens[:self.model_max_seq_len - 2]
        tokens=torch.Tensor(tokens).long()
        return tokens

    def __getitem__(self, i):
        if self.tok_type=='hf':
            #hugginface tokenizer
            tokens=self.tokenizer(self.dataset[i]["text"])['input_ids']
            # print(tokens)
        elif self.tok_type=='tokenmonster':
            s=self.norm.normalize_str(self.dataset[i]["text"])
            tokens = self.vocab.tokenize(s).tolist()
        tokens=tokens[:self.model_max_seq_len - 2]
        labels=tokens[1:]
        
        #casual language modelling: shift right by one
        tokens=tokens[:-1]
        l=len(tokens)
        for j in range(l,self.model_max_seq_len - 2):
            tokens.append(0)
            labels.append(0)
        tokens=torch.Tensor(tokens)
        labels=torch.Tensor(labels)

        input_ids=torch.as_tensor(tokens,dtype=torch.long)
        labels=torch.as_tensor(labels,dtype=torch.long)
        
        d={'input_ids':input_ids, 
            'labels':labels,
            'index':i,
        }
        return d
    
def divide_chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)] 

from random import shuffle

def texts2mlm(texts,domain,norm,vocab,MODEL_MAX_SEQ_LEN=84):
    input_ids=[]
    # token_type_ids=[]
    # attention_mask=[]
    # data=[]
    inputs=[]
    # masks=[]
    for t in texts:
        s=norm.normalize_str(t["text"])
        tokens = vocab.tokenize(s).tolist()
        
        #trucate
        tokens=tokens[:MODEL_MAX_SEQ_LEN - 2]
        l=len(tokens)
        for j in range(l,MODEL_MAX_SEQ_LEN - 2):
            tokens.append(0)

        tokens=torch.Tensor(tokens)
        if MODEL_MAX_SEQ_LEN - 2>l:
            att_mask=np.concatenate((np.ones(l),np.zeros(MODEL_MAX_SEQ_LEN - 2-l)))
        else:
            att_mask=np.ones(tokens.shape[0])
        # print(l,att_mask.shape[0],MODEL_MAX_SEQ_LEN - 2)
        assert tokens.shape[0]==att_mask.shape[0]

        input_ids=tokens

        input_ids=torch.as_tensor(input_ids,dtype=torch.long)
        inputs.append(input_ids)
    
    return {'input_ids':torch.stack(inputs),'domain':domain,'subdomain1':(domain*N_DOMAINS)+subdomain1}

class DomainDataset(torch.utils.data.Dataset):
    def __init__(self,dataset,cluster_labels,n_domains,norm,vocab,batch_size=5) -> None:
        self.dataset=dataset
        self.cluster_labels=cluster_labels
        self.bin_dataset={}
        self.domains=[i for i in range(n_domains)]
        self.batch_ordering=[]
        self.current_domain=0
        self.bs=batch_size
        self.norm=norm
        self.vocab=vocab

        self.fill_bins()

    def fill_bins(self):
        self.bin_dataset={}
        for i,c in enumerate(self.cluster_labels):
            if c not in self.bin_dataset:
                self.bin_dataset[c]=[i]
            else:
                self.bin_dataset[c].append(i)
        domains=[]
        for i in range(len(self.domains)):
            self.bin_dataset[i]=divide_chunks(self.bin_dataset[i],self.bs)
            for k in range(len(self.bin_dataset[i])):
                domains.append((i,k))
        shuffle(domains)
        self.batch_ordering=domains

    def __getitem__(self, i) -> torch.Tensor:
        indexes=self.bin_dataset[self.batch_ordering[i][0]][self.batch_ordering[i][1]]
        batch_data=[]
        for j in indexes:
            batch_data.append(self.dataset[j])
        # print('batch_data:',batch_data)
        batch_data=texts2mlm(batch_data,self.batch_ordering[i][0],self.norm,self.vocab)
        # batch_data = torch.from_numpy(a).long()
        return batch_data

    def __len__(self):
        #colocar uma margem de erro pra baixo
        return len(self.batch_ordering)


def get_vocab():
    #### TokenMonster BRRR!!!
    import tokenmonster
    vocab = tokenmonster.load("englishcode-16000-balanced-v1")
    # vocab = tokenmonster.load("englishcode-1024-balanced-v1")

    norm=normalizers.Sequence(
        [
            normalizers.Replace(Regex("(``|'')"), '"'),
            normalizers.NFD(),
            normalizers.Lowercase(),
            normalizers.StripAccents(),
            normalizers.Replace(Regex(" {2,}"), " "),
            normalizers.Replace(Regex(r"[^\x00-\x7F]+"), ""),
        ]
    )
    vocab.modify("[EOS]")
    vocab.modify("[UNK]")
    vocab.modify("[SEP]")
    vocab.modify("[PAD]")
    vocab.modify("[CLS]")
    vocab.modify("[MASK]")
    return norm,vocab


def get_tokenizer(dataset,num_tokenizer_train_items,vocab_size,tokenizer_path,model_training,model_max_seq_len,tok_type='hf'):
    
    #tokenmonster
    norm,vocab=get_vocab()
    if tok_type=='tokenmonster':
        MASK_ID=vocab.tokenize("[MASK]")[0]
        PAD_ID=vocab.tokenize("[PAD]")[0]
    elif tok_type=='hf':
        MASK_ID=103
        PAD_ID=0
   
    #tokenizer HF bert
    ## Train tokenizer
    tokenizer = BertWordPieceTokenizer()
    tokenizer._tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Replace(Regex("(``|'')"), '"'),
            normalizers.NFD(),
            normalizers.Lowercase(),
            normalizers.StripAccents(),
            normalizers.Replace(Regex(" {2,}"), " "),
            normalizers.Replace(Regex(r"[^\x00-\x7F]+"), ""),
        ]
    )  # Normalizer based on, https://github.com/JonasGeiping/cramming/blob/50bd06a65a4cd4a3dd6ee9ecce1809e1a9085374/cramming/data/tokenizer_preparation.py#L52
    with MagicTimer() as timer:
        tokenizer.train_from_iterator(
            tokenizer_training_data(dataset,num_tokenizer_train_items,len(dataset)),
            vocab_size=vocab_size,
            min_frequency=2,
        )
    print(f"Tokenizer trained in {timer}.")
    tokenizer.save(str(tokenizer_path))
    tokenizer = BertTokenizerFast(tokenizer_file=str(tokenizer_path))

    #load bert one
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    if model_training=='none':
        return dataset,norm,vocab
    
    elif model_training=='smlm':
        tokenized_dataset = TokenizedSMLMDataset(dataset,model_max_seq_len,norm,vocab,tokenizer,MASK_ID,PAD_ID,tok_type)
    elif model_training=='clm':
        tokenized_dataset = TokenizedCLMDataset(dataset,model_max_seq_len,norm,vocab,tokenizer,tok_type)
    elif model_training=='mlm':
        tokenized_dataset = TokenizedMLMDataset(dataset,model_max_seq_len,norm,vocab,tokenizer,MASK_ID,PAD_ID,tok_type)
    elif model_training=='domain':
        tokenized_dataset = DomainDataset(dataset,tokenizer, model_max_seq_len,norm,vocab)
   

    # print(tokenizer.mask_token,tokenizer.convert_tokens_to_ids(tokenizer.mask_token))

    # return tokenizer,tokenized_dataset,norm,vocab
    return tokenized_dataset,norm,vocab,tokenizer



