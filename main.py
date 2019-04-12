import torch
import torchtext
from torchtext.data import Field, BucketIterator
from torch import nn, optim

from AverageBaseline import AverageBaseline
from SNLIBatchGenerator import SNLIBatchGenerator


def get_accuracy(scores, true_labels):
    pred = torch.argmax(scores, dim=1)
    accuracy = torch.sum(pred == true_labels, dtype=torch.float32) / scores.shape[0]
    return accuracy


if __name__ == '__main__':

    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_epochs = 100
    # glove_file = 'F:\\Academics\\UvA\\Period 5\\SMNLS\\Practical\\InferSent\\.vector_cache\\glove.840B.300d.txt'

    tokenize = lambda x: x.split()
    TEXT = Field(sequential=True, tokenize=tokenize, lower=True, use_vocab=True, batch_first=False)
    LABEL = Field(sequential=False, use_vocab=True, pad_token=None, unk_token=None, batch_first=False)

    glove = torchtext.vocab.Vectors(name='glove.840B.300d.txt', max_vectors=100000)

    train, _, _ = torchtext.datasets.SNLI.splits(TEXT, LABEL)
    TEXT.build_vocab(train, vectors=glove)
    LABEL.build_vocab(train)

    train_iter = BucketIterator(dataset=train, batch_size=64, sort_key=None)
    batch_loader = SNLIBatchGenerator(train_iter)

    vocab_size = len(TEXT.vocab)
    baseline_model = AverageBaseline(vocab_size, 300, 512, 3, TEXT.vocab.vectors).to(device)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, baseline_model.parameters()), lr=0.1, weight_decay=0.99)
    cross_entropy_loss = nn.CrossEntropyLoss()

    for epoch in range(1, train_epochs + 1):
        loss_in_epoch = 0
        print("Epoch %d/%d:" % (epoch, train_epochs))
        for batch_id, (p, h, l) in enumerate(batch_loader):
            print("Step %d" % batch_id)
            premise = p.to(device)
            hypothesis = h.to(device)
            label = l.to(device)
            out = baseline_model(premise, hypothesis)
            loss = cross_entropy_loss(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_in_epoch += loss
        print("Loss = %f" % loss_in_epoch)