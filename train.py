import torch
import torchtext
from torchtext.data import Field, BucketIterator
from torch import nn, optim
import argparse
from SNLIBatchGenerator import SNLIBatchGenerator
from classifier import SNLIClassifier

# Default parameters
LEARNING_RATE_DEFAULT = 0.1
BATCH_SIZE_DEFAULT = 64
MAX_EPOCHS_DEFAULT = 100
GLOVE_SIZE_DEFAULT = 1000000
WEIGHT_DECAY_DEFAULT = 0.01


def get_accuracy(scores, true_labels):
    pred = torch.argmax(scores, dim=1)
    accuracy = torch.sum(pred == true_labels, dtype=torch.float32) / scores.shape[0]
    return accuracy


def train_model():

    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # glove_file = 'F:\\Academics\\UvA\\Period 5\\SMNLS\\Practical\\InferSent\\.vector_cache\\glove.840B.300d.txt'

    tokenize = lambda x: x.split()
    TEXT = Field(sequential=True, tokenize=tokenize, lower=True, use_vocab=True, batch_first=False)
    LABEL = Field(sequential=False, use_vocab=True, pad_token=None, unk_token=None, batch_first=False)

    glove_vectors = torchtext.vocab.Vectors(name='glove.840B.300d.txt', max_vectors=args.glove_size)

    train_set, valid_set, _ = torchtext.datasets.SNLI.splits(TEXT, LABEL)
    train_set.examples = train_set.examples[0:5000]
    valid_set.examples = valid_set.examples[0:5000]
    TEXT.build_vocab(train_set, valid_set, vectors=glove_vectors)
    LABEL.build_vocab(train_set)

    train_iter, valid_iter = BucketIterator.splits(datasets=(train_set, valid_set),
                                                   batch_sizes=(args.batch_size, args.batch_size),
                                                   sort_key=None,
                                                   device=device)

    train_batch_loader = SNLIBatchGenerator(train_iter)
    valid_batch_loader = SNLIBatchGenerator(valid_iter)

    vocab_size = len(TEXT.vocab)
    model = SNLIClassifier(encoder='average',
                           vocab_size=vocab_size,
                           embedding_dim=300,
                           hidden_dim=300,
                           fc_dim=512,
                           num_classes=3,
                           pretrained_vectors=TEXT.vocab.vectors).to(device)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.learning_rate, weight_decay=args.weight_decay)
    cross_entropy_loss = nn.CrossEntropyLoss()

    prev_valid_accuracy = 0
    finished_training = False
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        loss_in_epoch = 0
        train_accuracy = 0
        print("Epoch %d/%d:" % (epoch, args.max_epochs))
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


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_epochs', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Maximum number of epochs to train the model')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size for training the model')
  parser.add_argument('--glove_size', type=int, default=GLOVE_SIZE_DEFAULT,
                      help='Number of GloVe vectors to load initially')
  parser.add_argument('--weight_decay', type=int, default=WEIGHT_DECAY_DEFAULT,
                      help='Weight decay for the optimizer')
  args, unparsed = parser.parse_known_args()

  train_model()