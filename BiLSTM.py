import torch
from torch import nn


class BiLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super(BiLSTM, self).__init__()
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

    def forward(self, sentence_embed, sentence_len):
        _, (hidden_state, _) = self.encoder(sentence_embed)
        hidden_state = torch.cat((hidden_state[0], hidden_state[1]), dim=1)
        return hidden_state
