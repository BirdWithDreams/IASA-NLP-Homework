import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data import TextTokenDataset, pad_batch
from metrics import comp_metric

DEVICE = torch.device(['cpu', 'cuda'][torch.cuda.is_available()])


def find_connected_components(arr_list):
    components_list = []

    for arr in arr_list:
        if isinstance(arr, torch.Tensor):
            arr = arr.numpy()
        padded_arr = np.insert(arr, [0, arr.size], [False, False])
        diff = np.diff(padded_arr.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]  # Not subtracting 1 anymore

        components_list.append(list(zip(starts, ends)))

    return components_list


def get_precise_text_spans(original_text, tokens, token_spans):
    positions = []
    last_index = 0
    for token in tokens:
        start_idx = original_text.find(token, last_index)
        if start_idx == -1:
            token_stripped = token.strip('.,«»"()[]{}:;')
            start_idx = original_text.find(token_stripped, last_index)
            if start_idx == -1:
                raise ValueError(f"Token '{token}' not found in the original text after index {last_index}")
        end_idx = start_idx + len(token)
        positions.append((start_idx, end_idx))
        last_index = end_idx

    text_spans = []
    for start, end in token_spans:
        if start >= len(positions) or end > len(positions):
            raise ValueError(f"Invalid token span ({start}, {end}). Exceeds length of tokens.")
        text_spans.append((positions[start][0], positions[end - 1][1]))

    return text_spans


def predict_token_spans(model_class, state_dict, dataset, threshold=.5):
    from training import torch_loop

    model = model_class(**state_dict['model_params'])
    model.load_state_dict(state_dict['state_dict'])

    # dataset = TextTokenDataset(
    #     texts=df["text"].to_list(),
    #     loc_markers=df["loc_markers"].to_list(),
    #     vocab=vocab,
    #     tokenizer=tokenizer,
    # )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=pad_batch
    )

    labels, losses, targets, texts = torch_loop(
        dataloader=dataloader,
        inp_model=model,
        inp_optimizer=optim.Adam(model.parameters()),
        inp_criterion=nn.BCEWithLogitsLoss(reduction='none'),
        device=DEVICE,
        mode="eval",
        skip_concat=True
    )

    prediction_token_spans = find_connected_components([el.squeeze() > threshold for el in labels])
    return prediction_token_spans


def predict_text_spans(model_class, state_dict, df, vocab, tokenizer, threshold=0.5):
    dataset = TextTokenDataset(
        df['text'].tolist(),
        df['loc_markers'].tolist(),
        vocab,
        tokenizer,
    )

    prediction_token_spans = predict_token_spans(model_class, state_dict, dataset, threshold)

    prediction_text_spans = [
        get_precise_text_spans(
            text_sample,
            tokens,
            spans
        ) for text_sample, tokens, spans in zip(
            df['text'],
            [[token.text for token in tokenizer(sample)] for sample in df['text']],
            prediction_token_spans
        )
    ]

    return prediction_text_spans


def predict_locations(model_class, state_dict, df, vocab, tokenizer):
    from text import get_locations
    prediction_text_spans = predict_text_spans(model_class, state_dict, df, vocab, tokenizer)
    prediction_df = pd.DataFrame(
        {
            "text": df['text'],
            "loc_markers": prediction_text_spans
        }
    )

    real_locations = df.apply(get_locations, axis=1)
    prediction_locations = prediction_df.apply(get_locations, axis=1)

    return real_locations, prediction_locations


def show_predictions(model_class, state_dict, df, vocab, tokenizer):
    real_locations, prediction_locations = predict_locations(model_class, state_dict, df, vocab, tokenizer)

    print(
        "Holdout Test Metric:",
        comp_metric(
            real_locations.to_list(),
            prediction_locations
        ), '\n'
    )

    sorted_locations = np.argsort([len(el) for el in prediction_locations])

    for i in range(3):
        sample_id_with_many_locs = sorted_locations[-i - 1]

        print(
            "Text:", df.iloc[sample_id_with_many_locs].text,
            "\nReal locations:", real_locations[sample_id_with_many_locs],
            "\nPredicted locations:", prediction_locations[sample_id_with_many_locs],
            "\nSample Score:",
            comp_metric(
                [real_locations[sample_id_with_many_locs]], [prediction_locations[sample_id_with_many_locs]]
            ), '\n'
        )
