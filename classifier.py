import torch
from torch import nn

from AverageBaseline import AverageBaseline


class SNLIClassifier(nn.Module):

    def __init__(self, encoder, vocab_size, embedding_dim, hidden_dim, fc_dim, num_classes, pretrained_vectors):
        super(SNLIClassifier, self).__init__()
        if encoder == 'average':
            assert embedding_dim == hidden_dim
            self.encoder = AverageBaseline(vocab_size, embedding_dim, pretrained_vectors)
        else:
            raise ValueError('The encoder type is not supported')
        self.fc_dim = fc_dim
        self.num_classes = num_classes
        self.fc = nn.Sequential(nn.Linear(4 * hidden_dim, fc_dim),
                                nn.Tanh(),
                                # nn.Linear(fc_dim,fc_dim),
                                # nn.Tanh(),
                                nn.Linear(fc_dim, num_classes))

    def forward(self, s1, s2):
        u = self.encoder(s1)
        v = self.encoder(s2)
        abs_diff = torch.abs(u - v)
        mult = u * v
        feature_vector = torch.cat((u, v, abs_diff, mult), dim=1)
        out = self.fc(feature_vector)
        return out