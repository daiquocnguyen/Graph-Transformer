import os
import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
from tqdm import tqdm

"""
Modified from https://github.com/CRIPAC-DIG/TextING/blob/master/build_graph.py
"""

"""Sample: python build_graph.py mr 50 path/to/glove/"""

if len(sys.argv) < 4:
	sys.exit("Use: python build_graph.py <dataset> <word_embedding_dim> <pre_trained_folder>")

# settings
datasets = ['mr', 'ohsumed', 'R8', 'R52', 'TREC', 'ag_news', 'WebKB', 'SST1', 'SST2']

dataset = sys.argv[1]
if dataset not in datasets:
    sys.exit("wrong dataset name")


window_size = 3
print('using default window size = 3')

weighted_graph = False
print('using default unweighted graph')

truncate = False # whether to truncate long document
MAX_TRUNC_LEN = 350


print('loading raw data')

# load pre-trained word embeddings
word_embeddings_dim = int(sys.argv[2])
word_embeddings = {}
glove_path = str(sys.argv[3])
with open(glove_path + '/glove.6B.' + str(word_embeddings_dim) + 'd.txt', 'r') as f:
    for line in f.readlines():
        data = line.split()
        word_embeddings[str(data[0])] = list(map(float,data[1:]))


# load document list
doc_name_list = []
doc_train_list = []
doc_test_list = []

with open('data/' + dataset + '.txt', 'r') as f:
    for line in f.readlines():
        doc_name_list.append(line.strip())
        temp = line.split("\t")

        if temp[1].find('test') != -1:
            doc_test_list.append(line.strip())
        elif temp[1].find('train') != -1:
            doc_train_list.append(line.strip())
# map and shuffle
train_ids = []
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
random.shuffle(train_ids)

test_ids = []
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
random.shuffle(test_ids)

ids = train_ids + test_ids

# select 90% training set
train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size
test_size = len(test_ids)


# load raw text
doc_content_list = []
with open('data/corpus/' + dataset + '.clean.txt', 'r') as f:
    for line in f.readlines():
        doc_content_list.append(line.strip())

shuffle_doc_name_list = []
shuffle_doc_words_list = []
for i in ids:
    shuffle_doc_name_list.append(doc_name_list[int(i)])
    shuffle_doc_words_list.append(doc_content_list[int(i)])

# build corpus vocabulary
word_set = set()
for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    word_set.update(words)

vocab = list(word_set)
vocab_size = len(vocab)

word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

# initialize out-of-vocabulary word embeddings
oov = {}
for v in vocab:
    oov[v] = np.random.uniform(-0.01, 0.01, word_embeddings_dim)

# build label list
label_set = set()
for doc_meta in shuffle_doc_name_list:
    temp = doc_meta.split('\t')
    label_set.add(temp[2])
label_list = list(label_set)

# build graph function
def build_graph(start, end):
    x_adj = []
    x_feature = []
    y = []
    doc_len_list = []
    vocab_set = set()

    for i in tqdm(range(start, end)):
        doc_words = shuffle_doc_words_list[i].split()
        if truncate:
            doc_words = doc_words[:MAX_TRUNC_LEN]
        doc_len = len(doc_words)

        doc_vocab = list(set(doc_words))
        doc_nodes = len(doc_vocab)

        doc_len_list.append(doc_nodes)
        vocab_set.update(doc_vocab)

        doc_word_id_map = {}
        for j in range(doc_nodes):
            doc_word_id_map[doc_vocab[j]] = j

        # sliding windows
        windows = []
        if doc_len <= window_size:
            windows.append(doc_words)
        else:
            for j in range(doc_len - window_size + 1):
                window = doc_words[j: j + window_size]
                windows.append(window)

        word_pair_count = {}
        for window in windows:
            for p in range(1, len(window)):
                for q in range(0, p):
                    word_p = window[p]
                    word_p_id = word_id_map[word_p]
                    word_q = window[q]
                    word_q_id = word_id_map[word_q]
                    if word_p_id == word_q_id:
                        continue
                    word_pair_key = (word_p_id, word_q_id)
                    # word co-occurrences as weights
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.
                    # bi-direction
                    word_pair_key = (word_q_id, word_p_id)
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.

        row = []
        col = []
        weight = []
        features = []

        for key in word_pair_count:
            p = key[0]
            q = key[1]
            row.append(doc_word_id_map[vocab[p]])
            col.append(doc_word_id_map[vocab[q]])
            weight.append(word_pair_count[key] if weighted_graph else 1.)
        adj = sp.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))
        x_adj.append(adj)

        for k, v in sorted(doc_word_id_map.items(), key=lambda x: x[1]):
            features.append(word_embeddings[k] if k in word_embeddings else oov[k])
        x_feature.append(features)

    # labels
    for i in range(start, end):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        y.append(label_list.index(label))
    y = np.array(y)

    return x_adj, x_feature, y, doc_len_list, vocab_set


print('building graphs for training')
x_adj, x_feature, y, _, _ = build_graph(start=0, end=real_train_size)
print('building graphs for validation')
vx_adj, vx_feature, vy, _, _ = build_graph(start=real_train_size, end=train_size)
print('building graphs for test')
tx_adj, tx_feature, ty, _, _ = build_graph(start=train_size, end=train_size + test_size)

# dump objects
with open("data/ind.{}.x_adj".format(dataset), 'wb') as f:
    pkl.dump(x_adj, f)

with open("data/ind.{}.x_embed".format(dataset), 'wb') as f:
    pkl.dump(x_feature, f)

with open("data/ind.{}.y".format(dataset), 'wb') as f:
    pkl.dump(y, f)

with open("data/ind.{}.tx_adj".format(dataset), 'wb') as f:
    pkl.dump(tx_adj, f)

with open("data/ind.{}.tx_embed".format(dataset), 'wb') as f:
    pkl.dump(tx_feature, f)

with open("data/ind.{}.ty".format(dataset), 'wb') as f:
    pkl.dump(ty, f)

with open("data/ind.{}.vx_adj".format(dataset), 'wb') as f:
    pkl.dump(vx_adj, f)

with open("data/ind.{}.vx_embed".format(dataset), 'wb') as f:
    pkl.dump(vx_feature, f)

with open("data/ind.{}.vy".format(dataset), 'wb') as f:
    pkl.dump(vy, f)
