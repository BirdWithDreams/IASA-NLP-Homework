from copy import deepcopy

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import apply_connected_regions_and_compute_metric

DEVICE = torch.device(['cpu', 'cuda'][torch.cuda.is_available()])


def torch_loop(
        dataloader,
        inp_model,
        inp_optimizer,
        inp_criterion,
        mode="train",
        device="cpu",
        skip_concat=False,
        verbose=False,
):
    inp_model = inp_model.to(device)
    if mode == "train":
        inp_model.train()
    else:
        inp_model.eval()
    all_predicted_label = []
    all_text_ids = []
    all_losses = []
    all_targets = []
    with torch.inference_mode(mode=(mode != "train")):
        for text, label in tqdm(dataloader, disable=(verbose != 2)):
            text, label = text.to(device), label.to(device)
            if mode == "train":
                inp_optimizer.zero_grad()
            predicted_label = inp_model(text)
            loss = inp_criterion(predicted_label, label.float())
            if mode == "train":
                loss.mean().backward()
                inp_optimizer.step()

            all_predicted_label.extend(list(torch.sigmoid(predicted_label.detach()).cpu()))
            all_text_ids.extend(list(text.detach().cpu()))
            all_losses.extend(list(loss.detach().cpu()))
            all_targets.extend(list(label.detach().cpu()))

    if not skip_concat:
        from torch.nn.utils.rnn import pad_sequence
        all_losses = pad_sequence(all_losses, batch_first=True, padding_value=0)
        all_text_ids = pad_sequence(all_text_ids, batch_first=True, padding_value=1)
        all_predicted_label = pad_sequence(all_predicted_label, batch_first=True, padding_value=0)
        all_targets = pad_sequence(all_targets, batch_first=True, padding_value=0)

    return all_predicted_label, all_losses, all_targets, all_text_ids


def training_loop(
        epochs: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion,
        scheduler,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        device: torch.device = DEVICE,
        verbose=0,
):
    model = model.to(device)
    scheduler.verbose = (verbose > 0)

    train_all_epoch_losses = []
    valid_all_epoch_losses = []
    valid_metrics = []

    best_metric = {
        "f1": - np.inf,
    }
    best_model_state_dict = None

    for epoch in range(1, epochs + 1):
        if verbose >= 1:
            print(f"Starting Epoch {epoch}")

        if verbose == 2:
            print("Train phase")
        _, train_epoch_losses, _, _ = torch_loop(
            dataloader=train_dataloader,
            inp_model=model,
            inp_optimizer=optimizer,
            inp_criterion=criterion,
            device=device,
            mode="train",
            verbose=verbose,
        )

        if verbose == 2:
            print("Train BCE loss:", train_epoch_losses.mean())
            print("Valid phase")

        valid_epoch_labels, valid_epoch_losses, valid_epoch_targets, valid_epoch_texts = torch_loop(
            dataloader=test_dataloader,
            inp_model=model,
            inp_optimizer=optimizer,
            inp_criterion=criterion,
            device=device,
            mode="eval",
            verbose=verbose,
        )
        # 2.2 Compute and print valid metrics
        valid_metric = apply_connected_regions_and_compute_metric(
            valid_epoch_targets,
            valid_epoch_labels,
            valid_epoch_texts,
            tresh=0.5
        )
        if verbose == 2:
            print("Valid metric:", valid_metric)
            print("Valid BCE loss:", valid_epoch_losses.mean())
        # 3. Update learning rate (if needed)
        scheduler.step(valid_metric["f1"])
        # 4. Save best model
        if valid_metric["f1"] > best_metric["f1"]:
            best_metric = valid_metric
            state_dict = deepcopy(model.state_dict())
            best_model_state_dict = {
                'model_params': model.params,
                'state_dict': state_dict
            }

        # 5. Accumulate some stats
        train_all_epoch_losses.append(train_epoch_losses)
        valid_all_epoch_losses.append(valid_epoch_losses)
        valid_metrics.append(valid_metric)
    return best_model_state_dict, best_metric
