import torch
from torch import nn


class BiLSTMMaxPool(nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super(BiLSTMMaxPool, self).__init__()
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

    def forward(self, sentence_embed, sentence_len):
        all_states, _ = self.encoder(sentence_embed)
        max_out = torch.max(all_states, dim=0)[0]
        return max_out
