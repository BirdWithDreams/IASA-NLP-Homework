import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn, optim
from torch.utils.data import SubsetRandomSampler, DataLoader, Dataset

from data import TextTokenDataset, pad_batch, build_vocab
from training import training_loop, torch_loop
from metrics import apply_connected_regions_and_compute_metric

DEVICE = torch.device(['cpu', 'cuda'][torch.cuda.is_available()])


def cross_validate(
        model_class: nn.Module,
        model_params: dict,
        dataset: Dataset,
        stratification,
        k=5,
        epochs=10,
        learning_rate=0.001
):
    metrics = []
    best_models = []
    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    folds = StratifiedKFold(n_splits=k, shuffle=True)
    for fold, (train_ids, test_ids) in enumerate(folds.split([0] * len(dataset), stratification)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = SubsetRandomSampler(train_ids)
        valid_subsampler = SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        train_loader = DataLoader(
            dataset,
            batch_size=32,
            sampler=train_subsampler,
            drop_last=True,
            num_workers=0,
            collate_fn=pad_batch,
        )
        valid_loader = DataLoader(
            dataset,
            batch_size=64,
            sampler=valid_subsampler,
            num_workers=0,
            collate_fn=pad_batch,
        )

        # Init the neural network
        net = model_class(**model_params)  # replace with your network
        criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor([1]).view(1, 1, 1).to(DEVICE))
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=0, factor=0.5, min_lr=1e-7, mode="max", verbose=True
        )
        # Run the training loop for defined number of epochs
        best_model_state_dict, best_metric = training_loop(
            epochs,
            net,
            optimizer,
            criterion,
            scheduler,
            train_loader,
            valid_loader,
            verbose=1
        )
        metrics.append(best_metric)
        best_models.append(best_model_state_dict)
        # Process is complete.
        print('Training process has finished.')
    return metrics, best_models


def validate(
        model_class: nn.Module,
        model_params: dict,
        train_dataset: Dataset,
        test_dataset: Dataset,
        stratification,
        k=5,
        epochs=10,
        learning_rate=0.001
):
    metrics, best_models = cross_validate(
        model_class,
        model_params,
        train_dataset,
        stratification,
        k,
        epochs,
        learning_rate
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=0,
        collate_fn=pad_batch,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=pad_batch,
    )

    print(f'Average CV score: {np.mean([metric["f1"] for metric in metrics])}')
    print('Best models from validation scores: ', end='')
    test_metrics = []
    for state_dict in best_models:
        net = model_class(**state_dict['model_params'])
        net.load_state_dict(state_dict['state_dict'])

        criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor([1]).view(1, 1, 1).to(DEVICE))
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        test_epoch_labels, test_epoch_losses, test_epoch_targets, test_epoch_texts = torch_loop(
            dataloader=test_dataloader,
            inp_model=net,
            inp_optimizer=optimizer,
            inp_criterion=criterion,
            mode="eval",
            device=DEVICE
        )

        test_metric = apply_connected_regions_and_compute_metric(
            test_epoch_targets,
            test_epoch_labels,
            test_epoch_texts,
            tresh=0.5
        )
        test_metrics.append(test_metric)
        print(f'{np.round(test_metric["f1"], 4)} ', end='')

    print('\n'+'Training model for test'.center(50, '='))

    net = model_class(**state_dict['model_params'])
    net.load_state_dict(state_dict['state_dict'])

    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor([1]).view(1, 1, 1).to(DEVICE))
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=0, factor=0.5, min_lr=1e-7, mode="max", verbose=True
    )

    best_test_model_dict, best_test_score = (
        training_loop(
            epochs=epochs,
            model=net,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            verbose=2
        ))

    print(f'Model score on test dataset: {best_test_score}')

    return best_test_model_dict, best_test_score, *max(zip(best_models, test_metrics), key=lambda x: x[1]["f1"])


if __name__ == '__main__':
    import spacy
    import pandas as pd
    from models import UniversalRNN

    df = pd.read_csv('../datasets/small_dataset.csv', converters={"loc_markers": eval, "clean_loc_markers": eval})
    train_df = df[df['is_valid'] == 0]
    test_df = df[df['is_valid'] == 1]

    tokenizer = spacy.load("xx_ent_wiki_sm", disable=["tagger", "parser", "ner", "textcat"])
    vocab = build_vocab(train_df['clean_text'], tokenizer)

    train = TextTokenDataset(train_df['clean_text'].tolist(), train_df['clean_loc_markers'].tolist(), vocab, tokenizer)
    test = TextTokenDataset(test_df['clean_text'].tolist(), test_df['clean_loc_markers'].tolist(), vocab, tokenizer)

    params = {
        'num_embeddings': len(vocab),
        'out_channels': 1,
        'rnn_channels': 64,
        'num_layers': 8,
    }

    validate(
        UniversalRNN,
        params,
        train,
        test,
        train_df['stratify_col'].tolist(),
        3, 5
    )

