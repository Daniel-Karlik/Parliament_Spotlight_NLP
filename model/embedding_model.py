from typing import List, Union
import torch
import torch.nn as nn


class EmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, values: List[str], queries: Union[str, List[str]], embeddings, tokenizer, filter):
        if type(queries) == str:
            my_queries = queries
        else:
            my_queries = ' '.join(queries)
        query_encodings = embeddings(torch.tensor(tokenizer.encode(my_queries))).sum(dim=0)
        values_tokens = [tokenizer.encode(value) for value in values]
        values_encodings = [(filter(torch.tensor(value_tokens), embeddings) * embeddings(torch.tensor(value_tokens))).mean(dim=0) for value_tokens in values_tokens]
        values_matrix = torch.tensor(
            [
                [value_encoding_item for value_encoding_item in value_encodings]
                for value_encodings in values_encodings
            ]
        )

        query_encodings = query_encodings.detach()



        out = torch.matmul(values_matrix, query_encodings)
        return torch.nn.LogSoftmax(dim=0)(out.abs())
        # return out
