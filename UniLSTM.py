from torch import nn


class UniLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, pretrained_vectors):
        super(UniLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(pretrained_vectors)
        self.embedding.requires_grad = False
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, bidirectional=False)

    def forward(self, sentence, sentence_len):
        embed = self.embedding(sentence)
        _, (hidden_state, _) = self.encoder(embed)
        hidden_state.squeeze_(0)
        return hidden_state
