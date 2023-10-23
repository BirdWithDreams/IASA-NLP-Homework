import os
import datetime

import numpy as np
import spacy
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from models import UniversalRNN, ResRNN
from data import TextTokenDataset, build_vocab, pad_batch, DynamicTextTokenDataset, build_vocab_from_file
from validation import validate
from training import training_loop, DEVICE

tokenizer = spacy.load("xx_ent_wiki_sm", disable=["tagger", "parser", "ner", "textcat"])
text_column = 'hard_clean_text'
loc_column = 'hard_clean_loc_markers'

df = pd.read_csv(
    '../datasets/uk_ru(docs).csv',
    usecols=[text_column, loc_column, 'is_valid'],
    converters={loc_column: eval}
)

train_df = df[df['is_valid'] == 0]
test_df = df[df['is_valid'] == 1]


vocab = build_vocab(train_df[text_column], tokenizer)

num_workers = 0

train = TextTokenDataset(train_df[text_column].tolist(), train_df[loc_column].tolist(), vocab, tokenizer)
test = TextTokenDataset(test_df[text_column].tolist(), test_df[loc_column].tolist(), vocab, tokenizer)

# train_path = '../datasets/uk_ru(docs)_train.csv'
# test_path = '../datasets/uk_ru(docs)_test.csv'
#
# with open(train_path, 'r', encoding='utf-8') as f:
#     train_size = sum(1 for _ in f) - 1
#
# with open(test_path, 'r', encoding='utf-8') as f:
#     test_size = sum(1 for _ in f) - 1
#
# vocab = build_vocab_from_file(train_path, 'hard_clean_text', tokenizer)
#
# import sys
# print(sys.getsizeof(tokenizer))
# print(sys.getsizeof(vocab))
#
# train = DynamicTextTokenDataset(
#     '../datasets/uk_ru(docs)_train.csv',
#     vocab,
#     tokenizer,
#     'hard_clean_text',
#     'hard_clean_loc_markers',
#     train_size
#     )
#
# test = DynamicTextTokenDataset(
#     test_path,
#     vocab,
#     tokenizer,
#     'hard_clean_text',
#     'hard_clean_loc_markers',
#     test_size
# )

train_dataloader = DataLoader(
    train,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    num_workers=num_workers,
    collate_fn=pad_batch,
)

test_dataloader = DataLoader(
    test,
    batch_size=32,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=pad_batch,
)

models_params = [
    {
        'model': {
            'class': UniversalRNN,
            'params':
                {
                    'num_embeddings': len(vocab),
                    'out_channels': 1,
                    'rnn_channels': 64,
                    'num_layers': 4,
                },
        },

        'scheduler': {
            'class': optim.lr_scheduler.ReduceLROnPlateau,
            'params':
                {
                    'patience': 0,
                    'factor': 0.5,
                    'min_lr': 1e-7,
                    'mode': "max",
                    'verbose': True
                }
        },

        'epochs': 10,
    },
    {
        'model': {
            'class': UniversalRNN,
            'params':
                {
                    'num_embeddings': len(vocab),
                    'out_channels': 1,
                    'rnn_channels': 64,
                    'num_layers': 8,
                },
        },

        'scheduler': {
            'class': optim.lr_scheduler.ReduceLROnPlateau,
            'params':
                {
                    'patience': 0,
                    'factor': 0.5,
                    'min_lr': 1e-7,
                    'mode': "max",
                    'verbose': True
                }
        },

        'epochs': 10,
    },
    {
        'model': {
            'class': ResRNN,
            'params':
                {
                    'num_embeddings': len(vocab),
                    'out_channels': 1,
                    'rnn_channels': 64,
                    'num_layers': 4,
                },
        },

        'scheduler': {
            'class': optim.lr_scheduler.ReduceLROnPlateau,
            'params':
                {
                    'patience': 0,
                    'factor': 0.5,
                    'min_lr': 1e-7,
                    'mode': "max",
                    'verbose': True
                }
        },

        'epochs': 10,
    },
    {
        'model': {
            'class': ResRNN,
            'params':
                {
                    'num_embeddings': len(vocab),
                    'out_channels': 1,
                    'rnn_channels': 64,
                    'num_layers': 8,
                },
        },

        'scheduler': {
            'class': optim.lr_scheduler.ReduceLROnPlateau,
            'params':
                {
                    'patience': 0,
                    'factor': 0.5,
                    'min_lr': 1e-7,
                    'mode': "max",
                    'verbose': True
                }
        },

        'epochs': 10,
    },
    {
        'model': {
            'class': UniversalRNN,
            'params':
                {
                    'num_embeddings': len(vocab),
                    'out_channels': 1,
                    'rnn_channels': 64,
                    'num_layers': 8,
                },
        },

        'scheduler': {
            'class': optim.lr_scheduler.CosineAnnealingLR,
            'params':
                {
                    'T_max': 10,
                    'verbose': True
                }
        },

        'epochs': 10,
    },
    {
        'model': {
            'class': ResRNN,
            'params':
                {
                    'num_embeddings': len(vocab),
                    'out_channels': 1,
                    'rnn_channels': 64,
                    'num_layers': 8,
                },
        },

        'scheduler': {
            'class': optim.lr_scheduler.CosineAnnealingLR,
            'params':
                {
                    'T_max': 10,
                    'verbose': True
                }
        },

        'epochs': 10,
    },
]

if __name__ == '__main__':
    for params in models_params:
        time = datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S")
        path = f'../checkpoints/' + time + '/'
        os.mkdir(path)
        # best_test_model_dict, best_test_score, best_cv_model_dict, best_cv_score = validate(
        #     UniversalRNN,
        #     params,
        #     train,
        #     test,
        #     train_df['stratify_col'].tolist(),
        #     4, 5
        # )

        # torch.save(best_test_model_dict, path + f'Test model {np.round(best_test_score["f1"], 4)}.pth')
        # torch.save(best_cv_model_dict, path + f'CV model {np.round(best_cv_score["f1"], 4)}.pth')
        # torch.save(vocab, path + 'vocab.voc')

        model_dict = params['model']
        net = model_dict['class'](**model_dict['params'])

        criterion = nn.BCEWithLogitsLoss(
            reduction="none",
            pos_weight=torch.tensor([1]).view(1, 1, 1).to(DEVICE)
        )
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        scheduler_dict = params['scheduler']
        scheduler = scheduler_dict['class'](optimizer, **scheduler_dict['params'])

        best_state_dict, best_score, best_threshold = training_loop(
            epochs=params['epochs'],
            model=net,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            verbose=2
        )

        torch.save(best_state_dict, path + f'Model {np.round(best_score["f1"], 4)} {np.round(best_threshold, 4)}.pth')
        torch.save(vocab, path + 'vocab.voc')
