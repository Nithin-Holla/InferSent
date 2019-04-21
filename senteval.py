# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import logging
import argparse
import torch
import torchtext

from SNLIClassifier import SNLIClassifier

DEFAULT_SENTEVAL_PATH = '../SentEval'
DEFAULT_DATA_PATH = '../SentEval/data'

# import SentEval
sys.path.insert(0, DEFAULT_SENTEVAL_PATH)
import senteval


# SentEval prepare and batcher
def prepare(params, samples):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint_path)
    vocab = checkpoint['text_vocab']
    vocab_size = len(vocab)

    glove_vectors = torchtext.vocab.Vectors(name='senteval_glove.txt')
    unk_vector = torch.mean(glove_vectors.vectors, dim=0)

    model = SNLIClassifier(encoder=args.model_type,
                           vocab_size=vocab_size,
                           embedding_dim=300,
                           hidden_dim=2048,
                           fc_dim=512,
                           num_classes=3,
                           pretrained_vectors=None).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    params['device'] = device
    params['encoder'] = model.encoder
    params['vectors'] = glove_vectors
    params['unk_vector'] = unk_vector


def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    batch_size = len(batch)
    sent_lengths = [len(sentence) for sentence in batch]
    sent_lengths = torch.LongTensor(sent_lengths).to(params['device'])
    longest_length = torch.max(sent_lengths)

    word_embeddings = torch.ones((longest_length, batch_size, params['vectors'].dim)).to(params['device'])

    for sent_id, sent in enumerate(batch):
        for word_id, word in enumerate(sent):
            if isinstance(word, str):
                if word in params['vectors'].stoi:
                    word_embeddings[word_id, sent_id, :] = params['vectors'].vectors[params['vectors'].stoi[word]]
            elif isinstance(word, bytes):
                if word.decode('UTF-8') in params['vectors'].stoi:
                    word_embeddings[word_id, sent_id, :] = params['vectors'].vectors[params['vectors'].stoi[word.decode('UTF-8')]]
            else:
                word_embeddings[word_id, sent_id, :] = params['unk_vector']

    with torch.no_grad():
        sent_embeddings = params['encoder'](word_embeddings, sent_lengths)

    return sent_embeddings.cpu().numpy()


# Set params for SentEval
params_senteval = {'task_path': DEFAULT_DATA_PATH, 'usepytorch': True, 'kfold': 10}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', choices={'average', 'uniLSTM', 'biLSTM', 'biLSTMmaxpool'},
                        help='Type of encoder for the sentences')
    parser.add_argument('checkpoint_path', type=str,
                        help='Path to load the model checkpoint')
    parser.add_argument('--senteval_path', type=str, default=DEFAULT_SENTEVAL_PATH,
                        help='Path to SentEval repository')
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH,
                        help='Path to SentEval')
    args = parser.parse_args()

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC', 'SICKRelatedness',
                      'SICKEntailment', 'STS14', 'ImageCaptionRetrieval']
    results = se.eval(transfer_tasks)
    print(results)
