import torch
import torchtext
from torchtext.data import Field, BucketIterator
from torch import nn, optim

from AverageBaseline import AverageBaseline
from SNLIBatchGenerator import SNLIBatchGenerator
from UniLSTM import UniLSTM


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

    glove = torchtext.vocab.Vectors(name='glove.840B.300d.txt', max_vectors=1000000)
    # glove = torchtext.vocab.Vectors(name='small_glove_embed.npy')

    train_set, valid_set, _ = torchtext.datasets.SNLI.splits(TEXT, LABEL)
    train_set.examples = train_set.examples[0:5000]
    valid_set.examples = valid_set.examples[0:5000]
    TEXT.build_vocab(train_set, valid_set, vectors=glove)
    LABEL.build_vocab(train_set)

    train_iter, valid_iter = BucketIterator.splits(datasets=(train_set, valid_set),
                                                   batch_sizes=(64, 64),
                                                   sort_key=None,
                                                   device=device)

    train_batch_loader = SNLIBatchGenerator(train_iter)
    valid_batch_loader = SNLIBatchGenerator(valid_iter)

    vocab_size = len(TEXT.vocab)
    model = AverageBaseline(vocab_size, 300, 512, 3, TEXT.vocab.vectors).to(device)
    # model = UniLSTM(vocab_size, 300, 512, 3, TEXT.vocab.vectors).to(device)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1, weight_decay=0.01)
    cross_entropy_loss = nn.CrossEntropyLoss()

    prev_valid_accuracy = 0
    finished_training = False
    for epoch in range(1, train_epochs + 1):
        model.train()
        loss_in_epoch = 0
        train_accuracy = 0
        print("Epoch %d/%d:" % (epoch, train_epochs))
        for batch_id, (premise, hypothesis, label) in enumerate(train_batch_loader):
            out = model(premise, hypothesis)
            loss = cross_entropy_loss(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_in_epoch += loss.detach().item()
            train_accuracy += get_accuracy(out, label)
        loss_in_epoch /= batch_id
        train_accuracy /= batch_id

        model.eval()
        valid_accuracy = 0
        with torch.no_grad():
            for batch_id, (premise, hypothesis, label) in enumerate(valid_batch_loader):
                out = model(premise, hypothesis)
                valid_accuracy += get_accuracy(out, label)
            valid_accuracy /= batch_id
        print("train loss = %f, train accuracy = %f, valid accuracy = %f" % (loss_in_epoch, train_accuracy, valid_accuracy))

        if valid_accuracy <= prev_valid_accuracy:
            for param_group in optimizer.param_groups:
                if param_group['lr'] < 1e-5:
                    finished_training = True
                    break
                param_group['lr'] /= 5
        prev_valid_accuracy = valid_accuracy

        if finished_training:
            break

    print("Finished training")