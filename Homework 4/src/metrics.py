import re
from typing import List

import numpy as np
import pandas as pd


def hard_processing(text):
    text = re.sub(r"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\b\w\b\s?', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def comp_metric(y_true: List[List[str]], y_pred: List[List[str]]):
    assert len(y_true) == len(y_pred)
    tp, fp, fn, p, = 0.0, 0.0, 0.0, 0.0

    for y_true_sample, y_pred_sample in zip(y_true, y_pred):
        y_true_sample = set([hard_processing(s) for s in y_true_sample])
        y_pred_sample = set([hard_processing(s) for s in y_pred_sample])

        tp += len(y_true_sample & y_pred_sample)
        fp += len(y_pred_sample - y_true_sample)
        fn += len(y_true_sample - y_pred_sample)
        p += len(y_true_sample)

    if tp + fp == 0:
        if p == 0:
            precision = 1.0
        else:
            precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        if p == 0:
            recall = 1.0
        else:
            recall = 0.0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def extract_token_sequences(tensor, text_ids):
    extracted_texts = []

    # Add a False column at the beginning and end for detecting regions at boundaries
    tensor = np.column_stack((np.zeros(tensor.shape[0], dtype=bool), tensor, np.zeros(tensor.shape[0], dtype=bool)))

    for i, sequence in enumerate(tensor):
        # Find indices where values change
        diff = np.where(sequence[:-1] != sequence[1:])[0]

        sequence_texts = []
        for j in range(0, len(diff), 2):
            start, end = diff[j], diff[j + 1] - 1
            # Extract the token ids for the region and join them with "_"
            sequence_texts.append("_".join(map(str, text_ids[i, start:end + 1])))

        extracted_texts.append(sequence_texts)

    return extracted_texts


def apply_connected_regions_and_compute_metric(
        y_true,
        y_pred,
        texts,
        tresh=0.5
):
    real_tokens = extract_token_sequences(
        y_true[:, :, 0] > 0.5,
        texts
    )
    pred_tokens = extract_token_sequences(
        y_pred[:, :, 0] > tresh,
        texts
    )

    return comp_metric(real_tokens, pred_tokens)


def comp_model_metric(model_class, state_dict, df, vocab, tokenizer):
    from post_processing import predict_locations

    real_locations, prediction_locations = predict_locations(model_class, state_dict, df, vocab, tokenizer)

    return comp_metric(
        real_locations.to_list(),
        prediction_locations
    )
