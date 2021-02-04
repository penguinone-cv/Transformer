import csv
from tqdm import tqdm
import subprocess
import json
"""
csvファイルから各種パラメータを一括読み込みする関数
Argument
csv_path: csvファイルのパス
"""
def read_parameters(csv_path, index):
    with open(csv_path, encoding='utf-8-sig') as f: #utf-8-sigでエンコードしないと1列目のキーがおかしくなる
        reader = csv.DictReader(f)
        l = [row for row in tqdm(reader)]
        parameters_dict = l[index]

    return parameters_dict

#文字列のTrueをbool値のTrueに変換しそれ以外をFalseに変換する関数
def str_to_bool(str):
    return str.lower() == "true"
