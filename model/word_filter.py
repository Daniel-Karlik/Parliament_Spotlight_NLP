from pydoc import doc
from typing import Any, Dict, Union
import pickle

from transformers import ElectraForPreTraining, ElectraTokenizerFast
import torch
import torch.nn as nn
import numpy as np


class FilterBuilder:
    def __init__(self, path: str):
        self.path = path

    @staticmethod
    def escape(input: str) -> str:
        chars = [input_char.lower() for input_char in input if input_char not in '.,!?:()\'";[]\{\}\\/| ']
        return ''.join(chars)

    def word_buffer(self, document):
        text = document
        for line in text.split('\n'):
            for word in line.split(' '):
                escaped = FilterBuilder.escape(word)
                if escaped != '':
                    yield escaped
            
    def get_feature(self, counts, doc_count):
        feature = np.zeros((doc_count,))
        feature[0:len(counts)] += np.array(counts)
        return feature

    def build_features(self, limit: Union[int, None] = None) -> Dict[str, Dict[str, Any]]:
        features = {}
        data = None
        document_counter = 0
        word_counter = 0
        with open(self.path, 'rb') as picklefile:
            data = pickle.load(picklefile)
        for datapoint in data['results']:
            document_counter += 1
            text = datapoint['text']
            document_features = {}
            for word in self.word_buffer(text):
                word_counter += 1
                if word in document_features:
                    document_features[word] += 1
                else:
                    document_features[word] = 1
            for word in document_features:
                count = document_features[word]
                if word in features:
                    features[word].append(count / word_counter)
                else:
                    features[word] = [count / word_counter]
            word_counter = 0
            if limit is not None and document_counter == limit:
                break
        np_features = {word: self.get_feature(features[word], document_counter) for word in features}
        return np_features

    def get_nulls(self, limit: int = None):
        features = self.build_features(limit)
        labels = {}
        for word in features:
            feature = features[word]
            N = len(feature)
            nulls = len(feature[feature == 0])
            labels[word] = nulls / N
        return labels

    def paragraphs(self):
        data = None
        with open(self.path, 'rb') as picklefile:
            data = pickle.load(picklefile)
        for datapoint in data['results']:
            for paragraph in datapoint['text'].split('\n'):
                if paragraph != '':
                    yield paragraph


class WordFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(128, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16,1),
            nn.Sigmoid()
        )

    def forward(self, x, embeddings):
        word_embeddings = embeddings(x)
        # batch = word_embeddings.shape[0]
        return self.main(word_embeddings.detach())


class WordFilterLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class FilterBuilderWithTokenizer:
    def __init__(self, path: str, tokenizer):
        self.path = path
        self.tokenizer = tokenizer
            
    def get_feature(self, counts):
        feature = np.array(counts)
        return feature

    def build_features(self, limit: Union[int, None] = None) -> Dict[str, Dict[str, Any]]:
        features = {}
        data = None
        document_counter = 0
        word_counter = 0
        with open(self.path, 'rb') as picklefile:
            data = pickle.load(picklefile)
        for datapoint in data['results']:
            document_counter += 1
            text = datapoint['text']
            document_features = {}
            for word in self.tokenizer.encode(text, return_tensors='pt')[0]:
                word_counter += 1
                if word.item() in document_features:
                    document_features[word.item()] += 1
                else:
                    document_features[word.item()] = 1
            for word in document_features:
                count = document_features[word]
                if word in features:
                    features[word].append(count / word_counter)
                else:
                    features[word] = [count / word_counter]
            word_counter = 0
            if limit is not None and document_counter == limit:
                break
        np_features = {word: self.get_feature(features[word]) for word in features}
        return np_features, document_counter

    def get_nulls(self, limit: int = None):
        features, document_count = self.build_features(limit)
        labels = {}
        for word in features:
            feature = features[word]
            notnulls = len(feature)
            # if word.item() == 42:
            #     print(notnulls)
            labels[word] = (document_count - notnulls) / document_count
        return labels

    def paragraphs(self):
        data = None
        with open(self.path, 'rb') as picklefile:
            data = pickle.load(picklefile)
        for datapoint in data['results']:
            for paragraph in datapoint['text'].split('\n'):
                if paragraph != '':
                    yield paragraph
