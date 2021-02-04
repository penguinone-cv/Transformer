import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time, random
from torch.autograd import Variable
# For data loading.
from torchtext import data, datasets
from torchtext.vocab import Vectors
import re
import glob
import os
import io
import string


class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)

class Batch:
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


#?????????????????????????????????????????
def subsequent_mask(size):
    attention_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attention_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def preprocessing_text(text):
    # 改行コードを消去
    text = re.sub('<br />', '', text)

    # カンマ、ピリオド以外の記号をスペースに置換
    for p in string.punctuation:
        if (p == ".") or (p == ","):
            continue
        else:
            text = text.replace(p, " ")

    # ピリオドなどの前後にはスペースを入れておく
    text = text.replace(".", " . ")
    text = text.replace(",", " , ")
    return text

# 分かち書き（今回はデータが英語で、簡易的にスペースで区切る）


def tokenizer_punctuation(text):
    return text.strip().split()


# 前処理と分かち書きをまとめた関数を定義
def tokenizer_with_preprocessing(text):
    text = preprocessing_text(text)
    ret = tokenizer_punctuation(text)
    return ret

class DataLoader():
    def __init__(self, dataset_path, batch_size):
        self.max_length = 256
        self.TEXT = data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True,
                            lower=True, include_lengths=True, batch_first=True, fix_length=self.max_length, init_token="<cls>", eos_token="<eos>")
        self.LABEL = data.Field(sequential=False, use_vocab=False)

        self.train_val_ds, self.test_ds = data.TabularDataset.splits(
                                                path=dataset_path, train='IMDb_train.tsv',
                                                test='IMDb_test.tsv', format='tsv',
                                                fields=[('Text', self.TEXT), ('Label', self.LABEL)])

        self.train_ds, self.val_ds = self.train_val_ds.split(
                                                split_ratio=0.8, random_state=random.seed(1234))

        self.english_fasttext_vectors = Vectors(name='/home/data/IMDb/wiki-news-300d-1M.vec')
        self.TEXT.build_vocab(self.train_ds, vectors=self.english_fasttext_vectors, min_freq=10)
        self.train_dl = data.Iterator(self.train_ds, batch_size=batch_size, train=True)
        self.val_dl = data.Iterator(self.val_ds, batch_size=batch_size, train=False, sort=False)
        self.test_dl = data.Iterator(self.test_ds, batch_size=batch_size, train=False, sort=False)
