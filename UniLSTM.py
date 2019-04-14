import torch
from torch import nn


class UniLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, fc_dim, num_classes, pretrained_vectors):
        super(UniLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(pretrained_vectors)
        self.embedding.requires_grad = False
        self.num_classes = num_classes
        self.fc_dim = fc_dim
        self.encoder = nn.LSTM(embedding_dim, embedding_dim)
        self.fc = nn.Sequential(nn.Linear(4 * embedding_dim, fc_dim),
                                nn.ReLU(),
                                nn.Linear(fc_dim, num_classes))

    def forward(self, s1, s2):
        u = self.embedding(s1)
        _, u = self.encoder(u)
        v = self.embedding(s2)
        _, v = self.encoder(v)
        abs_diff = torch.abs(u - v)
        mult = u * v
        vector = torch.cat((u, v, abs_diff, mult), dim=1)
        out = self.fc(vector)
        return out