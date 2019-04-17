import torch
from torch import nn

class AverageBaseline(nn.Module):

    def __init__(self, vocab_size, embedding_dim, pretrained_vectors):
        super(AverageBaseline, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(pretrained_vectors)
        self.embedding.requires_grad = False

    def forward(self, sentence, sentence_len):
        embed = self.embedding(sentence)
        out = torch.sum(embed, dim=0) / sentence_len.unsqueeze(1).float()
        return out