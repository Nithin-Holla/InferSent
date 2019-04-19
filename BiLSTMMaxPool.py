import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BiLSTMMaxPool(nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super(BiLSTMMaxPool, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

    def forward(self, sentence_embed, sentence_len):
        sorted_lengths, sort_indices = torch.sort(sentence_len, descending=True)
        sentence_embed = sentence_embed[:, sort_indices, :]
        packed_seq = pack_padded_sequence(sentence_embed, sorted_lengths, batch_first=False)
        all_states, _ = self.encoder(packed_seq)
        pad_packed_states, _ = pad_packed_sequence(all_states, batch_first=False)
        _, unsorted_indices = torch.sort(sort_indices)
        pad_packed_states = pad_packed_states[:, unsorted_indices, :]
        flipped_states = torch.cat((pad_packed_states[:, :, 0:self.hidden_dim],
                                    torch.flip(pad_packed_states[:, :, self.hidden_dim:], dims=[2])), dim=2)
        max_out, _ = torch.max(flipped_states, dim=0)
        return max_out
