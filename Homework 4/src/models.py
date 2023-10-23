import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence


class UniversalRNN(nn.Module):

    def __init__(
            self,
            num_embeddings,
            out_channels,
            rnn_channels=512,
            num_layers=1,
            bidirectional=True,
            dropout_rate=0.4
    ):
        super().__init__()

        self.params = {
            'num_embeddings': num_embeddings,
            'out_channels': out_channels,
            'rnn_channels': rnn_channels,
            'num_layers': num_layers,
            'bidirectional': bidirectional,
            'dropout_rate': dropout_rate,
        }

        self.embedding_layer = nn.Sequential(
            nn.Embedding(num_embeddings, rnn_channels),
            nn.Dropout(dropout_rate / 2),
        )

        self.rnns = nn.GRU(
            rnn_channels,
            rnn_channels,
            bidirectional=bidirectional,
            num_layers=num_layers,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(
                rnn_channels * 2 if bidirectional else rnn_channels,
                out_channels,
            )
        )

    def forward(self, x):
        if isinstance(x, PackedSequence):
            emb = self.apply_embeddings(x)
        else:
            emb = self.embedding_layer(x)

        x = self.rnns(emb)[0]
        if isinstance(x, PackedSequence):
            x, _ = pad_packed_sequence(x, batch_first=True)

        x = self.classifier(x)
        return x

    def apply_embeddings(self, packed_indices: PackedSequence) -> PackedSequence:
        sequences, lengths = pad_packed_sequence(packed_indices, batch_first=True)
        sequences = self.embedding_layer(sequences)
        packed_embeddings = pack_padded_sequence(sequences, lengths, batch_first=True)
        return packed_embeddings


class ResRNN(nn.Module):

    def __init__(
            self,
            num_embeddings,
            out_channels,
            rnn_channels=512,
            num_layers=2,
            bidirectional=True,
            skip_embeddings=True,
            dropout_rate=0.4
    ):
        super().__init__()

        self.params = {
            'num_embeddings': num_embeddings,
            'out_channels': out_channels,
            'rnn_channels': rnn_channels,
            'num_layers': num_layers,
            'bidirectional': bidirectional,
            'skip_embeddings': skip_embeddings,
            'dropout_rate': dropout_rate,
        }

        self.bidirectional = bidirectional
        self.skip_embeddings = skip_embeddings

        self.embedding_layer = nn.Sequential(
            nn.Embedding(num_embeddings, rnn_channels * 2 if bidirectional else rnn_channels),
            nn.Dropout(dropout_rate / 2),
        )

        self.rnns = nn.ModuleList(
            [
                nn.GRU(
                    rnn_channels * 2 if bidirectional else rnn_channels,
                    rnn_channels,
                    num_layers=1,
                    bidirectional=bidirectional,
                    batch_first=True
                ) for _ in range(num_layers)
            ]
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(
                rnn_channels * 2 if bidirectional else rnn_channels,
                out_channels,
            )
        )

    def forward(self, x):
        emb = self.embedding_layer(x)

        x = self._apply_rnn(emb)

        if self.skip_embeddings:
            x = x + emb

        x = self.classifier(x)
        return x

    def _apply_rnn(self, x):
        x = self.rnns[0](x)[0]
        for i, rnn in enumerate(self.rnns[1:]):
            x = rnn(x)[0] + x

        return x



if __name__ == '__main__':
    x = torch.randint(0, 16, (8, 10))
    n = ResRNN(16, 1, 64, bidirectional=True)
    print(n(x).shape)
