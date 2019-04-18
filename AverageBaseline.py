import torch
from torch import nn

class AverageBaseline(nn.Module):

    def __init__(self):
        super(AverageBaseline, self).__init__()

    def forward(self, sentence_embed, sentence_len):
        out = torch.sum(sentence_embed, dim=0) / sentence_len.unsqueeze(1).float()
        return out