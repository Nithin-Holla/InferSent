# Import libraries
import os
import torch
import torchtext
from torchtext.data import Field, BucketIterator
from torch import nn, optim
import argparse
from tensorboardX import SummaryWriter
from SNLIBatchGenerator import SNLIBatchGenerator
from SNLIClassifier import SNLIClassifier
from nltk import word_tokenize

# Default parameters
LEARNING_RATE_DEFAULT = 0.1
BATCH_SIZE_DEFAULT = 64
MAX_EPOCHS_DEFAULT = 50
GLOVE_SIZE_DEFAULT = None
WEIGHT_DECAY_DEFAULT = 0


def get_accuracy(scores, true_labels):
    """
    Get the accuracy
    :param scores: Matrix of prediction scores of shape no. of batches x no. of classes
    :param true_labels: Vector of true labels of shape no. of batches
    :return:
    """
    pred = torch.argmax(scores, dim=1)
    accuracy = torch.sum(pred == true_labels, dtype=torch.float32) / scores.shape[0]
    return accuracy


def train_model():
    """
    Train the classifier model
    :return:
    """
    # Flags for deterministic runs
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # TensorboardX writer object
    tb_writer = SummaryWriter(os.path.join('logs', args.model_type))

    # Define fields for reading SNLI data
    TEXT = Field(sequential=True, tokenize=word_tokenize, lower=True, use_vocab=True, batch_first=False, include_lengths=True)
    LABEL = Field(sequential=False, use_vocab=True, pad_token=None, unk_token=None, batch_first=False)

    # Load GloVe vectors
    glove_vectors = torchtext.vocab.Vectors(name=args.vector_file, max_vectors=args.glove_size)

    # Load the training set and validation set
    train_set, valid_set, _ = torchtext.datasets.SNLI.splits(TEXT, LABEL, root=args.data_path)

    # Build the text and label vocabulary
    TEXT.build_vocab(train_set, valid_set, vectors=glove_vectors)
    LABEL.build_vocab(train_set)
    vocab_size = len(TEXT.vocab)

    # Set the vector for '<unk>' token as mean of other vectors
    TEXT.vocab.vectors[TEXT.vocab.stoi['<unk>']] = torch.mean(TEXT.vocab.vectors, dim=0)

    # Define the iterator over the train and valid set
    train_iter, valid_iter = BucketIterator.splits(datasets=(train_set, valid_set),
                                                   batch_sizes=(args.batch_size, args.batch_size),
                                                   sort_key=lambda x: x.premise,
                                                   shuffle=True,
                                                   sort_within_batch=True,
                                                   device=device)
    # Custom wrapper over the iterators
    train_batch_loader = SNLIBatchGenerator(train_iter)
    valid_batch_loader = SNLIBatchGenerator(valid_iter)

    # Define the model, the optimizer and the loss module
    model = SNLIClassifier(encoder=args.model_type,
                           vocab_size=vocab_size,
                           embedding_dim=300,
                           hidden_dim=2048,
                           fc_dim=512,
                           num_classes=3,
                           pretrained_vectors=TEXT.vocab.vectors).to(device)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.learning_rate, weight_decay=args.weight_decay)
    cross_entropy_loss = nn.CrossEntropyLoss()

    # Load the checkpoint if found
    if os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print("Resuming training from epoch %d with loaded model and optimizer..." % start_epoch)
    else:
        start_epoch = 1
        print("Training the model from scratch...")

    # Begin training
    prev_valid_accuracy = 0
    terminate_training = False
    for epoch in range(start_epoch, args.max_epochs + 1):
        model.train()
        train_loss = 0
        train_accuracy = 0
        print("Epoch %d:" % epoch)
        for batch_id, (premise, hypothesis, label) in enumerate(train_batch_loader):
            out = model(premise[0], hypothesis[0], premise[1], hypothesis[1])
            loss = cross_entropy_loss(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()
            train_accuracy += get_accuracy(out, label)
        train_loss /= batch_id
        train_accuracy /= batch_id
        tb_writer.add_scalar('train loss', train_loss, epoch)
        tb_writer.add_scalar('train accuracy', train_accuracy, epoch)

        # Evaluate the model on the validation set
        model.eval()
        valid_accuracy = 0
        with torch.no_grad():
            for batch_id, (premise, hypothesis, label) in enumerate(valid_batch_loader):
                out = model(premise[0], hypothesis[0], premise[1], hypothesis[1])
                valid_accuracy += get_accuracy(out, label)
            valid_accuracy /= batch_id
        tb_writer.add_scalar('valid accuracy', valid_accuracy, epoch)
        print("train loss = %f, train accuracy = %f, valid accuracy = %f" % (
            train_loss, train_accuracy, valid_accuracy))

        # Save the state and the vocabulary
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'text_vocab': TEXT.vocab.stoi,
            'label_vocab': LABEL.vocab.stoi
        }, args.checkpoint_path)

        # If validation accuracy does not improve, divide the learning rate by 5 and
        # if learning rate falls below 1e-5 terminate training
        if valid_accuracy <= prev_valid_accuracy:
            for param_group in optimizer.param_groups:
                if param_group['lr'] < 1e-5:
                    terminate_training = True
                    break
                param_group['lr'] /= 5
        prev_valid_accuracy = valid_accuracy
        if terminate_training:
            break

    # Termination message
    if terminate_training:
        print("Training terminated because the learning rate fell below %f" % 1e-5)
    else:
        print("Maximum epochs reached. Finished training")

    tb_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', choices={'average', 'uniLSTM', 'biLSTM', 'biLSTMmaxpool'},
                        help='Type of encoder for the sentences')
    parser.add_argument('checkpoint_path', type=str,
                        help='Path to save/load the checkpoint data')
    parser.add_argument('data_path', type=str,
                        help='Path where data is saved')
    parser.add_argument('vector_file', type=str,
                        help='File in which vectors are saved')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Maximum number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size for training the model')
    parser.add_argument('--glove_size', type=int,
                        help='Number of GloVe vectors to load initially')
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY_DEFAULT,
                        help='Weight decay for the optimizer')
    args = parser.parse_args()

    train_model()
