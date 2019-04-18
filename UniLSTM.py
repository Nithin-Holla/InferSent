from torch import nn
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class UniLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super(UniLSTM, self).__init__()
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, bidirectional=False)

    def forward(self, sentence_embed, sentence_len):
        sorted_lengths, sort_indices = torch.sort(sentence_len, descending=True)
        sentence_embed = sentence_embed[:, sort_indices, :]
        packed_seq = pack_padded_sequence(sentence_embed, sorted_lengths, batch_first=False)
        out, _ = self.encoder(packed_seq)
        unpacked_out = pad_packed_sequence(out, batch_first=False)
        final_hidden_state = unpacked_out[0][-1, :, :]
        return final_hidden_state
