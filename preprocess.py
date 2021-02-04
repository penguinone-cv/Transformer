# tsv形式のファイルにします
import glob
import os
import io
import string

import sys, codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout)

# 訓練データのtsvファイルを作成します

f = open('./IMDb_train.tsv', 'w', encoding='utf-8')

path = '/home/data/IMDb/aclImdb/train/pos/'
for fname in glob.glob(os.path.join(path, '*.txt')):
    with io.open(fname, 'r', encoding="utf-8") as ff:
        text = ff.readline()

        # タブがあれば消しておきます
        text = text.replace('\t', " ")

        text = text+'\t'+'1'+'\t'+'\n'
        f.write(text)

path = '/home/data/IMDb/aclImdb/train/neg/'
for fname in glob.glob(os.path.join(path, '*.txt')):
    with io.open(fname, 'r', encoding="utf-8") as ff:
        text = ff.readline()

        # タブがあれば消しておきます
        text = text.replace('\t', " ")

        text = text+'\t'+'0'+'\t'+'\n'
        f.write(text)

f.close()


# テストデータの作成

f = open('./IMDb_test.tsv', 'w', encoding='utf-8')

path = '/home/data/IMDb/aclImdb/test/pos/'
for fname in glob.glob(os.path.join(path, '*.txt')):
    with io.open(fname, 'r', encoding="utf-8") as ff:
        text = ff.readline()

        # タブがあれば消しておきます
        text = text.replace('\t', " ")

        text = text+'\t'+'1'+'\t'+'\n'
        f.write(text)


path = '/home/data/IMDb/aclImdb/test/neg/'

for fname in glob.glob(os.path.join(path, '*.txt')):
    with io.open(fname, 'r', encoding="utf-8") as ff:
        text = ff.readline()

        # タブがあれば消しておきます
        text = text.replace('\t', " ")

        text = text+'\t'+'0'+'\t'+'\n'
        f.write(text)

f.close()
