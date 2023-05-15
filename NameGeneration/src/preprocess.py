"""
    Created by @namhainguyen2803 in 10/05/2023
"""

import re
import numpy as np
import pandas as pd
import torch

DISTINCT_CHARACTER = ['UNK', ' ', 'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                      's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def convert(text):
    patterns = {
        '[àáảãạăắằẵặẳâầấậẫẩ]': 'a',
        '[đ]': 'd',
        '[èéẻẽẹêềếểễệ]': 'e',
        '[ìíỉĩị]': 'i',
        '[òóỏõọôồốổỗộơờớởỡợ]': 'o',
        '[ùúủũụưừứửữự]': 'u',
        '[ỳýỷỹỵ]': 'y'
    }
    output = text
    for regex, replace in patterns.items():
        output = re.sub(regex, replace, output)
        # deal with upper case
        output = re.sub(regex.upper(), replace.upper(), output)
    return output.lower()


def retrieve_word(document):
    list_doc = list()
    for name in document:
        list_word = convert(name)
        list_doc.append(list_word)
    return list_doc


def retrieve_character_from_word(word):
    return [*word]


def retrieve_character_from_document(list_document):
    list_doc = list()
    for i in range(len(list_document)):
        doc = list_document[i]
        normalized_doc = convert(doc)
        list_word = retrieve_character_from_word(normalized_doc)
        list_word.append("UNK")
        list_doc.append(list_word)
    return list_doc


def create_dictionary(list_split_character):
    set_word = set()
    for document in list_split_character:
        set_word = set_word.union(set(document))
    return set_word


def encode_word_from_document(list_split_character, character_to_index, max_character=12):
    if max_character == None:
        res = list()
        for i in range(len(list_split_character)):
            split_character = list_split_character[i]
            encode_char = np.zeros((len(split_character),))
            for j in range(len(split_character)):
                ind = character_to_index.get(split_character[j], -1)
                if ind != -1:
                    encode_char[j] = ind
            encode_char = torch.from_numpy(encode_char).to(torch.int64)
            res.append(encode_char)
        return res
    else:
        num_document = len(list_split_character)
        list_index_all = torch.zeros(num_document, max_character, dtype=torch.long)
        for i in range(len(list_split_character)):
            split_character = list_split_character[i]
            for j in range(len(split_character)):
                if j >= max_character:
                    break
                else:
                    ind = character_to_index.get(split_character[j], -1)
                    if ind != -1:
                        list_index_all[i, j] = ind
        return list_index_all


def decode_word(vect, distinct_character=DISTINCT_CHARACTER):
    list_word = list()
    for i in range(len(vect)):
        list_word.append(distinct_character[vect[i]])
    return list_word


def retrieve_dataset(parent_path):
    parent_path = "dataset"
    training_path = parent_path + "/name_train.csv"
    dev_path = parent_path + "/name_dev.csv"
    test_path = parent_path + "/name_test.csv"
    data_path = parent_path + "/name_full.csv"
    df = pd.read_csv(data_path)
    df_train = pd.read_csv(training_path)
    df_test = pd.read_csv(test_path)
    df_dev = pd.read_csv(dev_path)
    return df, df_train, df_test, df_dev


def create_character_to_token(dictionary=DISTINCT_CHARACTER):
    character_to_index = dict()
    for c in range(len(dictionary)):
        character_to_index[dictionary[c]] = c
    return character_to_index


def retrieve_dataset_from_path(data_path):
    res = list()
    with open(data_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            res.append(line.strip())
        return res


def retrieve_dataset_from_dataframe(data_path):
    return pd.read_csv(data_path)


def retrieve_middle_last_name_from_list_name(list_name):
    res = set()
    for i in range(len(list_name)):
        name = convert(list_name[i])
        list_word = name.split(" ")
        if len(list_word) <= 2:
            res.add(name)
        else:
            middle_last_name = " ".join((list_word[-2], list_word[-1]))
            res.add(middle_last_name)
    return list(res)
