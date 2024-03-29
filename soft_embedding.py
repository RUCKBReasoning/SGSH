import torch
import torch.nn as nn

class SoftEmbedding(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 10, 
                random_range: float = 0.5,
                initialize_from_vocab: bool = True):
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte, n_tokens, random_range, initialize_from_vocab))
            
    def initialize_embedding(self, wte: nn.Embedding, n_tokens: int = 10, random_range: float = 0.5, initialize_from_vocab: bool = True):
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.randn(n_tokens, wte.weight.size(1))
            
    def forward(self, tokens):
        n_tokens = torch.sum(tokens[0] == 50264).item()
        input_embedding = self.wte(tokens[:, :(tokens.size(1)-n_tokens)])
        learned_embedding = self.learned_embedding[:n_tokens, :].repeat(input_embedding.size(0), 1, 1)
        return torch.cat([input_embedding, learned_embedding], 1)
