import torchtext
from torchtext.data import Field, BucketIterator

from AverageBaseline import AverageBaseline
from SNLIBatchGenerator import SNLIBatchGenerator

if __name__ == '__main__':

    epochs = 10
    glove_file = 'F:\\Academics\\UvA\\Period 5\\SMNLS\\Practical\\InferSent\\.vector_cache\\glove.840B.300d.txt'

    tokenize = lambda x: x.split()
    TEXT = Field(sequential=True, tokenize=tokenize, lower=True, use_vocab=True, batch_first=True)
    LABEL = Field(sequential=False, use_vocab=True, pad_token=None, unk_token=None, batch_first=True)

    glove = torchtext.vocab.Vectors(name=glove_file, max_vectors=10000)

    _, train, _ = torchtext.datasets.SNLI.splits(TEXT, LABEL)
    TEXT.build_vocab(train, vectors=glove)
    LABEL.build_vocab(train)

    train_iter = BucketIterator(dataset=train, batch_size=32, sort_key=None)
    batch_loader = SNLIBatchGenerator(train_iter)

    vocab_size = len(TEXT.vocab)
    baseline_model = AverageBaseline(vocab_size, 300, 512, 3, TEXT.vocab.vectors)

    for epoch in range(1, epochs + 1):
        premise, hypothesis, label = next(iter(batch_loader))
        out = baseline_model(premise, hypothesis)
        print(out)