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
    eval_period = 10
    # glove_file = 'F:\\Academics\\UvA\\Period 5\\SMNLS\\Practical\\InferSent\\.vector_cache\\glove.840B.300d.txt'

    tokenize = lambda x: x.split()
    TEXT = Field(sequential=True, tokenize=tokenize, lower=True, use_vocab=True, batch_first=False)
    LABEL = Field(sequential=False, use_vocab=True, pad_token=None, unk_token=None, batch_first=False)

    glove = torchtext.vocab.Vectors(name='glove.840B.300d.txt', max_vectors=100000)

    train_set, valid_set, _ = torchtext.datasets.SNLI.splits(TEXT, LABEL)
    # train_set.examples = train_set.examples[0:5000]
    # valid_set.examples = valid_set.examples[0:5000]
    TEXT.build_vocab(train_set, valid_set, vectors=glove)
    LABEL.build_vocab(train_set)

    train_iter, valid_iter = BucketIterator.splits(datasets=(train_set, valid_set),
                                                   batch_sizes=(64, 64),
                                                   sort_key=None,
                                                   device=device)

    train_batch_loader = SNLIBatchGenerator(train_iter)
    valid_batch_loader = SNLIBatchGenerator(valid_iter)

    vocab_size = len(TEXT.vocab)
    baseline_model = AverageBaseline(vocab_size, 300, 512, 3, TEXT.vocab.vectors).to(device)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, baseline_model.parameters()), lr=0.0001, weight_decay=0.99)
    cross_entropy_loss = nn.CrossEntropyLoss()

    for epoch in range(1, train_epochs + 1):
        loss_in_epoch = 0
        train_accuracy = 0
        print("Epoch %d/%d:" % (epoch, train_epochs))
        for batch_id, (premise, hypothesis, label) in enumerate(train_batch_loader):
            out = baseline_model(premise, hypothesis)
            loss = cross_entropy_loss(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_in_epoch += loss
            train_accuracy += get_accuracy(out, label)
        loss_in_epoch /= batch_id
        train_accuracy /= batch_id

        if epoch % eval_period == 0:
            valid_accuracy = 0
            with torch.no_grad():
                for batch_id, (premise, hypothesis, label) in enumerate(valid_batch_loader):
                    out = baseline_model(premise, hypothesis)
                    valid_accuracy += get_accuracy(out, label)
                valid_accuracy /= batch_id
            print("Train accuracy = %f, valid accuracy = %f" %(train_accuracy, valid_accuracy))
        else:
            print("Train accuracy = %f" % train_accuracy)