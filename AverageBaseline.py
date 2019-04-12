import torch
from torch import nn

class AverageBaseline(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, pretrained_vectors):
        super(AverageBaseline, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(pretrained_vectors)
        self.embedding.requires_grad = False
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.fc = nn.Sequential(nn.Linear(4 * embedding_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, num_classes))

    def forward(self, s1, s2):
        embed_u = self.embedding(s1)
        embed_v = self.embedding(s2)
        avg_embed_u = torch.mean(embed_u, dim=0)
        avg_embed_v = torch.mean(embed_v, dim=0)
        abs_diff = torch.abs(avg_embed_u - avg_embed_v)
        mult = avg_embed_u * avg_embed_v
        vector = torch.cat((avg_embed_u, avg_embed_v, abs_diff, mult), dim=1)
        out = self.fc(vector)
        return out