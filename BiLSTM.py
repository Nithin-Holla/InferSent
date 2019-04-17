import torch
from torch import nn


class BiLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, pretrained_vectors):
        super(BiLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(pretrained_vectors)
        self.embedding.requires_grad = False
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

    def forward(self, sentence, sentence_len):
        embed = self.embedding(sentence)
        _, (hidden_state, _) = self.encoder(embed)
        hidden_state = torch.cat((hidden_state[0], hidden_state[1]), dim=1)
        return hidden_state
