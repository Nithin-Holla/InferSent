import torch
from torch import nn

from AverageBaseline import AverageBaseline
from BiLSTMMaxPool import BiLSTMMaxPool
from UniLSTM import UniLSTM
from BiLSTM import BiLSTM


class SNLIClassifier(nn.Module):

    def __init__(self, encoder, vocab_size, embedding_dim, hidden_dim, fc_dim, num_classes, pretrained_vectors):
        super(SNLIClassifier, self).__init__()
        if encoder == 'average':
            self.encoder = AverageBaseline(vocab_size, embedding_dim, pretrained_vectors)
            self.hidden_dim = embedding_dim
        elif encoder == 'uniLSTM':
            self.encoder = UniLSTM(vocab_size, embedding_dim, hidden_dim, pretrained_vectors)
            self.hidden_dim = hidden_dim
        elif encoder == 'biLSTM':
            self.encoder = BiLSTM(vocab_size, embedding_dim, hidden_dim, pretrained_vectors)
            self.hidden_dim = 2 * hidden_dim
        elif encoder == 'biLSTMmaxpool':
            self.encoder = BiLSTMMaxPool(vocab_size, embedding_dim, hidden_dim, pretrained_vectors)
            self.hidden_dim = 2 * hidden_dim
        else:
            raise ValueError('The encoder type is not supported')
        self.fc_dim = fc_dim
        self.num_classes = num_classes
        self.fc = nn.Sequential(nn.Linear(4 * self.hidden_dim, self.fc_dim),
                                nn.Tanh(),
                                nn.Linear(self.fc_dim, self.num_classes))

    def forward(self, s1, s2, s1_len, s2_len):
        u = self.encoder(s1, s1_len)
        v = self.encoder(s2, s2_len)
        feature_vector = torch.cat((u, v, torch.abs(u - v), u * v), dim=1)
        out = self.fc(feature_vector)
        return out
