import torch
from torch import nn


class BiLSTMMaxPool(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, pretrained_vectors):
        super(BiLSTMMaxPool, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(pretrained_vectors)
        self.embedding.requires_grad = False
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

    def forward(self, sentence):
        embed = self.embedding(sentence)
        all_states, _ = self.encoder(embed)
        max_out = torch.max(all_states, dim=0)[0]
        return max_out
