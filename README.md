# InferSent

Implementation of Conneau et al, [Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/abs/1705.02364).

## Dependencies
- Python 3
- PyTorch
- Torchtext
- NLTK
- TensorboardX
- Scikit-learn

## Training

```
usage: train.py [-h] [--learning_rate LEARNING_RATE] [--max_epochs MAX_EPOCHS]
                [--batch_size BATCH_SIZE] [--glove_size GLOVE_SIZE]
                [--weight_decay WEIGHT_DECAY]
                model_type checkpoint_path
                data_path vector_file

positional arguments:
  model_type            Type of encoder for the sentences (one of {biLSTMmaxpool,average,uniLSTM,biLSTM})
  checkpoint_path       Path to save/load the checkpoint data
  data_path             Path where data is saved
  vector_file           File in which vectors are saved

optional arguments:
  -h, --help            show this help message and exit
  --learning_rate       Learning rate
  --max_epochs          Maximum number of epochs to train the model
  --batch_size          Batch size for training the model
  --glove_size          Number of GloVe vectors to load initially
  --weight_decay        Weight decay for the optimizer
```

## Evaluating on the test set
```
usage: eval.py [-h] [--batch_size BATCH_SIZE]
               model_type checkpoint_path
               data_path

positional arguments:
  model_type            Type of encoder for the sentences (one of {biLSTMmaxpool,average,uniLSTM,biLSTM})
  checkpoint_path       Path to save/load the checkpoint data
  data_path             Path where data is saved

optional arguments:
  -h, --help            show this help message and exit
  --batch_size          Batch size for training the model
```

## Evaluating on SentEval
```
usage: senteval.py [-h] [--senteval_path SENTEVAL_PATH]
                   [--data_path DATA_PATH]
                   model_type checkpoint_path
                   vector_file

positional arguments:
  model_type            Type of encoder for the sentences (one of {biLSTMmaxpool,average,uniLSTM,biLSTM})
  checkpoint_path       Path to load the model checkpoint
  vector_file           File in which vectors are saved

optional arguments:
  -h, --help            show this help message and exit
  --senteval_path       Path to SentEval repository
  --data_path           Path to SentEval data
```

## Inference

```
usage: infer.py [-h] model_type checkpoint_path
                input_file output_file

positional arguments:
  model_type            Type of encoder for the sentences (one of {biLSTMmaxpool,average,uniLSTM,biLSTM})
  checkpoint_path       Path to save/load the checkpoint data
  input_file            Input JSON file containing premise-hypothesis pairs
  output_file           Output JSON file to write predictions to

optional arguments:
  -h, --help            show this help message and exit
```