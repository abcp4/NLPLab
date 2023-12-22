import pickle
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from kmeans_pytorch import KMeans as BalancedKMeans
import numpy as np
from tqdm.auto import tqdm
import torch
# !git clone https://github.com/kernelmachine/balanced-kmeans/
# !cd balanced-kmeans && pip install --editable .
# !pip install numba

def get_texts(dataset):
    return[dataset[i]['text'] for i in range(len(dataset))]

def number_normalizer(tokens):
    """Map all numeric tokens to a placeholder.

    For many applications, tokens that begin with a number are not directly
    useful, but the fact that such a token exists can be relevant.  By applying
    this form of dimensionality reduction, some methods may perform better.
    """
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


class NumberNormalizingVectorizer(TfidfVectorizer):
    # this vectorizer replaces numbers with #NUMBER token
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))


def train_vectorizer(texts):
    # english stopwords plus the #NUMBER dummy token
    stop_words = list(text.ENGLISH_STOP_WORDS.union(["#NUMBER"]))

    model = Pipeline([('tfidf', NumberNormalizingVectorizer(stop_words=stop_words)),
                      ('svd', TruncatedSVD(n_components=100)),
                      ('normalizer', Normalizer(copy=False))])

    model.fit(tqdm(texts[:1000000]))
    vecs = model.transform(tqdm(texts))
    return model, vecs

import math

def clusterize(vecs, N_DOMAINS=2):
    if torch.cuda.is_available():
            device = torch.device('cuda')
    else:
            device = torch.device('cpu')

    kmeans = BalancedKMeans(n_clusters=N_DOMAINS, device=device, balanced=True)

    bs=500
    batches = np.array_split(vecs, bs, axis=0)
    len(batches),batches[0].shape
    for i, batch in tqdm(enumerate(batches)):
            kmeans.fit(torch.from_numpy(batch), iter_limit=20, online=True, tqdm_flag=False,iter_k=i)
    #LEVEL 0
    cluster_labels=[]
    for i, batch in tqdm(enumerate(batches)):
        cluster_ids_y = kmeans.predict(
            X=torch.from_numpy(batch).to(device),tqdm_flag=False
        )
        for c in cluster_ids_y:
            cluster_labels.append(c.item())
    cluster_labels=np.asarray(cluster_labels)
    return cluster_labels

#Domain index(level 0),text index(from txts and tfid vecs), subdomain index(level 1)
# idxs_cluster_labels_level1[-3:]
# len(idxs_cluster_labels_level1)