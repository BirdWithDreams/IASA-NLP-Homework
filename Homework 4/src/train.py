import os
import datetime

import numpy as np
import spacy
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from models import UniversalRNN
from data import TextTokenDataset, build_vocab, pad_batch
from validation import validate
from training import training_loop, DEVICE

df = pd.read_csv(
    '../datasets/medium_dataset.csv',
    converters={"loc_markers": eval, "clean_loc_markers": eval}
)

train_df = df[df['is_valid'] == 0]
test_df = df[df['is_valid'] == 1]

tokenizer = spacy.load("xx_ent_wiki_sm", disable=["tagger", "parser", "ner", "textcat"])
vocab = build_vocab(train_df['clean_text'], tokenizer)

train = TextTokenDataset(train_df['clean_text'].tolist(), train_df['clean_loc_markers'].tolist(), vocab, tokenizer)
test = TextTokenDataset(test_df['clean_text'].tolist(), test_df['clean_loc_markers'].tolist(), vocab, tokenizer)

train_dataloader = DataLoader(
    train,
    batch_size=128,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    num_workers=0,
    collate_fn=pad_batch,
)

test_dataloader = DataLoader(
    test,
    batch_size=32,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    pin_memory=True,
    collate_fn=pad_batch,
)

model_params = [
    {
        'num_embeddings': len(vocab),
        'out_channels': 1,
        'rnn_channels': 64,
        'num_layers': 4,
    },
    {
        'num_embeddings': len(vocab),
        'out_channels': 1,
        'rnn_channels': 64,
        'num_layers': 8,
    },
    {
        'num_embeddings': len(vocab),
        'out_channels': 1,
        'rnn_channels': 64,
        'num_layers': 4,
        'bidirectional': False
    },
    {
        'num_embeddings': len(vocab),
        'out_channels': 1,
        'rnn_channels': 64,
        'num_layers': 8,
        'bidirectional': False
    },
    {
        'num_embeddings': len(vocab),
        'out_channels': 1,
        'rnn_channels': 128,
        'num_layers': 2,
    },
    {
        'num_embeddings': len(vocab),
        'out_channels': 1,
        'rnn_channels': 128,
        'num_layers': 4,
    },
    {
        'num_embeddings': len(vocab),
        'out_channels': 1,
        'rnn_channels': 128,
        'num_layers': 8,
    },
    {
        'num_embeddings': len(vocab),
        'out_channels': 1,
        'rnn_channels': 256,
        'num_layers': 4,
    },
]

if __name__ == '__main__':
    for params in model_params:
        # best_test_model_dict, best_test_score, best_cv_model_dict, best_cv_score = validate(
        #     UniversalRNN,
        #     params,
        #     train,
        #     test,
        #     train_df['stratify_col'].tolist(),
        #     4, 5
        # )

        time = datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S")
        path = f'../checkpoints/' + time + '/'
        os.mkdir(path)
        #
        # torch.save(best_test_model_dict, path + f'Test model {np.round(best_test_score["f1"], 4)}.pth')
        # torch.save(best_cv_model_dict, path + f'CV model {np.round(best_cv_score["f1"], 4)}.pth')
        # torch.save(vocab, path + 'vocab.voc')
        net = UniversalRNN(**params)
        criterion = nn.BCEWithLogitsLoss(
            reduction="none",
            pos_weight=torch.tensor([1]).view(1, 1, 1).to(DEVICE)
        )
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=0, factor=0.5, min_lr=1e-7, mode="max", verbose=True
        )

        best_state_dict, best_score = training_loop(
            epochs=8,
            model=net,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            verbose=2
        )

        torch.save(best_state_dict, path + f'Model {np.round(best_score["f1"], 4)}.pth')
        torch.save(vocab, path + 'vocab.voc')
