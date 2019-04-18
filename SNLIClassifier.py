import torch
from torch import nn

from AverageBaseline import AverageBaseline
from BiLSTMMaxPool import BiLSTMMaxPool
from UniLSTM import UniLSTM
from BiLSTM import BiLSTM


class SNLIClassifier(nn.Module):

    def __init__(self, encoder, vocab_size, embedding_dim, hidden_dim, fc_dim, num_classes, pretrained_vectors):
        super(SNLIClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_vectors is not None:
            self.embedding.weight.data.copy_(pretrained_vectors)
        self.embedding.requires_grad = False
        if encoder == 'average':
            self.encoder = AverageBaseline()
            self.hidden_dim = embedding_dim
        elif encoder == 'uniLSTM':
            self.encoder = UniLSTM(embedding_dim, hidden_dim)
            self.hidden_dim = hidden_dim
        elif encoder == 'biLSTM':
            self.encoder = BiLSTM(embedding_dim, hidden_dim)
            self.hidden_dim = 2 * hidden_dim
        elif encoder == 'biLSTMmaxpool':
            self.encoder = BiLSTMMaxPool(embedding_dim, hidden_dim)
            self.hidden_dim = 2 * hidden_dim
        else:
            raise ValueError('The encoder type is not supported')
        self.fc_dim = fc_dim
        self.num_classes = num_classes
        self.fc = nn.Sequential(nn.Linear(4 * self.hidden_dim, self.fc_dim),
                                nn.Tanh(),
                                nn.Linear(self.fc_dim, self.num_classes))

    def forward(self, s1, s2, s1_len, s2_len):
        s1_embed = self.embedding(s1)
        u = self.encoder(s1_embed, s1_len)
        s2_embed = self.embedding(s2)
        v = self.encoder(s2_embed, s2_len)
        feature_vector = torch.cat((u, v, torch.abs(u - v), u * v), dim=1)
        out = self.fc(feature_vector)
        return out
