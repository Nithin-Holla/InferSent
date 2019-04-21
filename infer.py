import torch
import argparse
import json
from nltk import word_tokenize

from SNLIClassifier import SNLIClassifier


def numericalize(sentence, text_vocab, device):
    sentence = word_tokenize(sentence)
    sentence_len = torch.LongTensor(1).to(device)
    sentence_len[0] = len(sentence)
    numerical_sent = [text_vocab[word] if word in text_vocab else 0 for word in sentence]
    numerical_sent = torch.tensor(numerical_sent).view(-1, 1).to(device)
    return numerical_sent, sentence_len


def infer():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint_path)
    text_vocab = checkpoint['text_vocab']
    label_vocab = checkpoint['label_vocab']
    label_mapping = sorted(label_vocab, key=label_vocab.get)
    vocab_size = len(text_vocab)

    model = SNLIClassifier(encoder=args.model_type,
                           vocab_size=vocab_size,
                           embedding_dim=300,
                           hidden_dim=2048,
                           fc_dim=512,
                           num_classes=3,
                           pretrained_vectors=None).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    with open(args.input_file, 'r') as f:
        example_list = json.load(f)

    results = []

    model.eval()
    with torch.no_grad():
        for example in example_list:
            premise, premise_len = numericalize(example['premise'], text_vocab, device)
            hypothesis, hypothesis_len = numericalize(example['hypothesis'], text_vocab, device)
            out = model(premise, hypothesis, premise_len, hypothesis_len)
            prediction = torch.argmax(out, dim=1)
            results.append({'premise': example['premise'],
                            'hypothesis': example['hypothesis'],
                            'prediction': label_mapping[prediction]})

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', choices={'average', 'uniLSTM', 'biLSTM', 'biLSTMmaxpool'},
                        help='Type of encoder for the sentences')
    parser.add_argument('checkpoint_path', type=str,
                        help='Path to save/load the checkpoint data')
    parser.add_argument('input_file', type=str,
                        help='Input file containing premise-hypothesis pairs')
    parser.add_argument('output_file', type=str,
                        help='Output file to write predictions to')
    args = parser.parse_args()

    infer()



