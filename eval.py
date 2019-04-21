import os
import torch
import torchtext
from torchtext.data import Field, Iterator
import argparse
from SNLIBatchGenerator import SNLIBatchGenerator
from SNLIClassifier import SNLIClassifier
from nltk import word_tokenize

# Default parameters
BATCH_SIZE_DEFAULT = 64


def get_accuracy(scores, true_labels):
    pred = torch.argmax(scores, dim=1)
    accuracy = torch.sum(pred == true_labels, dtype=torch.float32) / scores.shape[0]
    return accuracy


def eval_model():
    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # tokenize = lambda x: x.split()
    TEXT = Field(sequential=True, tokenize=word_tokenize, lower=True, use_vocab=True, batch_first=False, include_lengths=True)
    LABEL = Field(sequential=False, use_vocab=True, pad_token=None, unk_token=None, batch_first=False)

    _, _, test_set = torchtext.datasets.SNLI.splits(TEXT, LABEL)

    TEXT.build_vocab(test_set)
    LABEL.build_vocab(test_set)

    checkpoint = torch.load(args.checkpoint_path)
    TEXT.vocab.stoi = checkpoint['text_vocab']
    LABEL.vocab.stoi = checkpoint['label_vocab']

    vocab_size = len(TEXT.vocab.stoi)

    test_iter = Iterator(dataset=test_set, batch_size=args.batch_size, device=device)

    test_batch_loader = SNLIBatchGenerator(test_iter)

    # Define the model and the loss module
    model = SNLIClassifier(encoder=args.model_type,
                           vocab_size=vocab_size,
                           embedding_dim=300,
                           hidden_dim=2048,
                           fc_dim=512,
                           num_classes=3,
                           pretrained_vectors=None).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    with torch.no_grad():
        test_accuracy = 0
        with torch.no_grad():
            for batch_id, (premise, hypothesis, label) in enumerate(test_batch_loader):
                out = model(premise[0], hypothesis[0], premise[1], hypothesis[1])
                test_accuracy += get_accuracy(out, label)
            test_accuracy /= batch_id

    print("Test accuracy = %f" % test_accuracy)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', choices={'average', 'uniLSTM', 'biLSTM', 'biLSTMmaxpool'},
                        help='Type of encoder for the sentences')
    parser.add_argument('checkpoint_path', type=str,
                        help='Path to save/load the checkpoint data')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size for training the model')
    args = parser.parse_args()

    eval_model()
