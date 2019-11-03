import os
import numpy as np
import fileinput
from typing import List, Tuple, Dict
from cortecx.construction import tools as t
from cortecx.construction.foundation import DataObj
from keras.utils import to_categorical


def load_conll(file_path: str, task: str) -> Tuple:
    data = DataObj()
    data.retrieve(path=os.path.relpath(file_path), format=task)
    return data.data


def load_word_embeddings(file_path: str, limit):
    word_embeddings = {}
    line_tracker = 0
    for line in fileinput.input([file_path]):
        if line_tracker > limit:
            fileinput.close()
            break
        else:
            splits = str(line).replace('\n', '').split(' ')
            wrd = splits[0]
            vector = [float(num) for num in splits[1:]]
            word_embeddings.update({wrd: vector})
            line_tracker += 1
            continue
    return word_embeddings


def load_char_embeddings(file_path: str, limit):
    char_embeddings = {}
    line_tracker = 0
    for line in fileinput.input([file_path]):
        if line_tracker > limit:
            fileinput.close()
            break
        else:
            splits = str(line).replace('\n', '').split(' ')
            wrd = splits[0]
            vector = [float(num) for num in splits[1:]]
            char_embeddings.update({wrd: vector})
            line_tracker += 1
            continue
    return char_embeddings


def word_vectorize(text: str, vector_matrix: Dict) -> List:
    vectorized = []

    tokenizer = t.Tokenizer(text=text)
    tokens = tokenizer.tokenize()
    tokens = tokens.tokens

    for word in tokens:
        try:
            vectorized.append(vector_matrix[word])
        except KeyError:
            vectorized.append(np.zeros(300).tolist())
    return vectorized


def char_vectorize(word: str, vector_matrix: Dict) -> List:
    return [vector_matrix[letter] for letter in word]


def for_model_char_side(data, limit=1000000, pad_len=60) -> List:
    char_matrix = load_char_embeddings(os.path.relpath('./data/char_vectors.txt'), limit=limit)
    encoded = []
    for sentence in data[0]:
        for i, word in enumerate(sentence):
            sentence[i] = t.padding(char_vectorize(word[0], char_matrix), pad_len=10, pad_char=np.zeros(300).tolist())
        sentence = t.padding(sentence, pad_len=pad_len, pad_char=np.zeros(shape=(10, 300)).tolist())
        encoded.append(sentence)
    return encoded


def for_model_word_side(data, limit=10, pad_len=60) -> List:
    vector_matrix = load_word_embeddings(os.path.relpath('./data/word_vectors.txt'), limit=limit)
    encoded = []
    for element in data[0]:
        temp = []
        for point in element:
            try:
                temp.append(vector_matrix[point[0]])
            except KeyError:
                temp.append(np.zeros(300).tolist())
            temp = t.padding(temp, pad_len=pad_len, pad_char=np.zeros(300).tolist())
        encoded.append(temp)
    return encoded


def tag_output(data, tags_dict: Dict, num_classes=45, pad_len=60) -> List:
    tags = data[1]
    for i, tag in enumerate(tags):
        tags[i] = [tags_dict[element] for element in tag]
        tags[i] = t.padding(tags[i], pad_len=pad_len, pad_char=0)
    return [to_categorical(i, num_classes=num_classes) for i in tags]
