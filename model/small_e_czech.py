from typing import List, Union
from transformers import ElectraForPreTraining, ElectraTokenizerFast
import torch
import torch.nn as nn


class SmolicekEmbedingModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.w_q = nn.Linear(128, 300, bias=False)
        self.w_v = nn.Linear(128, 300, bias=False)


    def forward(self, values: List[str], queries: Union[str, List[str]], embeddings, tokenizer):
        if type(queries) == str:
            my_queries = queries
        else:
            my_queries = ' '.join(queries)
        query_encodings = embeddings(torch.tensor(tokenizer.encode(my_queries))).sum(dim=0)
        values_tokens = [tokenizer.encode(value) for value in values]
        values_encodings = [embeddings(torch.tensor(value_tokens)).mean(dim=0) for value_tokens in values_tokens]
        values_matrix = torch.tensor(
            [
                [value_encoding_item for value_encoding_item in value_encodings]
                for value_encodings in values_encodings
            ]
        )

        query_encodings = query_encodings.detach()



        out = torch.matmul(self.w_v(values_matrix), self.w_q(query_encodings))
        return torch.nn.LogSoftmax(dim=0)(out)


class TransformerSmolicek:
    def __init__(self):
        self.model = ElectraForPreTraining.from_pretrained('Seznam/small-e-czech')
        self.tokenizer = ElectraTokenizerFast.from_pretrained("Seznam/small-e-czech")

    def __call__(self,  values: List[str], queries: Union[str, List[str]]):
        if type(queries) == str:
            my_queries = queries
        else:
            my_queries = ' '.join(queries)
        query_tokens = self.tokenizer.encode(my_queries)
        values_tokens = [self.tokenizer.encode(value) for value in values]
        
        query_embeddings = self.model.electra.embeddings(query_tokens)
        values_embeddings = self.model.electra.embeddings(values_tokens)

        query_embeddings = self.model.electra.embeddings_project(query_embeddings)
        values_embeddings = self.model.electra.embeddings_project(values_embeddings)



class SmolicekEmbedingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_projection = nn.Linear(128, 256)
        self.w_v = nn.Linear(128, 300, bias=False)


    def forward(self, values: List[str], queries: Union[str, List[str]], small_e_czech):
        if type(queries) == str:
            my_queries = queries
        else:
            my_queries = ' '.join(queries)
        query_tokens = self.tokenizer.encode(my_queries)
        values_tokens = [self.tokenizer.encode(value) for value in values]

        
        query_embeddings = self.electra.embeddings_project(query_embeddings)
        

