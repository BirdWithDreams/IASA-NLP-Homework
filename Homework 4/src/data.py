import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import build_vocab_from_iterator

TEXT_PAD_IDX = 1
TARGET_PAD_IDX = 0


class TextTokenDataset(Dataset):
    def __init__(self, texts, loc_markers, vocab, tokenizer):
        assert len(texts) == len(loc_markers)
        self.texts = texts
        self.loc_markers = loc_markers
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __getitem__(self, item):
        text, locs = self.texts[item], self.loc_markers[item]
        tokens, labels = self.form_labeling(text, locs)

        tokens = self.vocab(tokens)
        tokens = torch.LongTensor(tokens)
        labels = torch.LongTensor(labels)
        return tokens, labels

    def __len__(self):
        return len(self.texts)

    def form_labeling(self, text, loc_marker):
        tokens = list(self.tokenizer(text))

        tokens_text = [token.text for token in tokens]
        labels = [0] * len(tokens)

        for idx, token in enumerate(tokens):
            for start, end in loc_marker:
                if token.idx >= start and (token.idx + len(token.text)) <= end:
                    labels[idx] = 1
                    break

        return tokens_text, labels


class DynamicTextTokenDataset(TextTokenDataset):
    def __init__(self, csv_file, vocab, tokenizer, text_column='clean_text', loc_markers_column='clean_loc_markers'):
        self.csv_file = csv_file

        with open(csv_file, 'r', encoding='utf-8') as data_io:
            self.column_names = data_io.readline().split(',')
            self.data_len = sum(1 for line in data_io)  # subtract 1 for header

        self.vocab = vocab
        self.tokenizer = tokenizer
        self.loc_markers_column = loc_markers_column
        self.text_column = text_column

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = pd.read_csv(
            self.csv_file,
            names=self.column_names,
            skiprows=1 + idx,
            nrows=1,
            converters={self.loc_markers_column: eval}
        )
        sample = data.iloc[0]

        text, locs = sample[self.text_column], sample[self.loc_markers_column]
        tokens, labels = self.form_labeling(text, locs)

        tokens = self.vocab(tokens)
        tokens = torch.LongTensor(tokens)
        labels = torch.LongTensor(labels)
        return tokens, labels


    # def form_labeling(self, text, loc_marker):
    #     tokens = list(self.tokenizer(text))
    #
    #     tokens_text = [token.text for token in tokens]
    #     labels = [0] * len(tokens)
    #
    #     for idx, token in enumerate(tokens):
    #         for start, end in loc_marker:
    #             if token.idx >= start and (token.idx + len(token.text)) <= end:
    #                 labels[idx] = 1
    #                 break
    #
    #     return tokens_text, labels


def form_dataloaders(train, valid, vocab, tokenizer):
    train_dataset = TextTokenDataset(
        train["text"].to_list(),
        train["clean_loc_makers"].to_list(),
        vocab,
        tokenizer,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=pad_batch,
    )

    valid_dataset = TextTokenDataset(
        valid["text"].to_list(),
        valid["clean_loc_makers"].to_list(),
        vocab,
        tokenizer,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=pad_batch,
    )
    return train_dataloader, valid_dataloader


def build_vocab(series, tokenizer):
    def yield_tokens(texts):
        for text in texts:
            yield [token.text for token in tokenizer(text)]

    vocab = build_vocab_from_iterator(
        yield_tokens(series.tolist()),
        specials=["<unk>", "<pad>"]
    )
    vocab.set_default_index(vocab["<unk>"])

    return vocab


def build_vocab_from_file(file_name, text_column, tokenizer, chunksize=500):
    def yield_tokens(texts):
        for text in texts:
            yield [token.text for token in tokenizer(text)]

    def yield_with_chunks():
        chunks = pd.read_csv(file_name, usecols=[text_column], chunksize=chunksize)
        for chunk in chunks:
            yield from yield_tokens(chunk.iloc[:, 0].tolist())

    vocab = build_vocab_from_iterator(
        yield_with_chunks(),
        specials=["<unk>", "<pad>"]
    )
    vocab.set_default_index(vocab["<unk>"])

    return vocab


def pack_batch(pair):
    x, y = list(zip(*pair))
    x = torch.nn.utils.rnn.pack_sequence(x, enforce_sorted=False)
    y = torch.nn.utils.rnn.pack_sequence(y, enforce_sorted=False)
    return x, y


def pad_batch(pair):
    x, y = list(zip(*pair))
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=TEXT_PAD_IDX)
    y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=TARGET_PAD_IDX)
    return x, y.unsqueeze(2)


if __name__ == '__main__':
    import spacy
    tokenizer = spacy.load("xx_ent_wiki_sm", disable=["tagger", "parser", "ner", "textcat"])
    vocab = build_vocab_from_file('../datasets/medium_dataset.csv', 'clean_text', tokenizer)
    print(len(vocab))
    dataset = DynamicTextTokenDataset('../datasets/medium_dataset.csv', vocab, tokenizer)
    print(dataset[10])
