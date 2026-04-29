import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class TicketLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        self.regressor = nn.Linear(hidden_dim * 2, 1)

    def forward(self, X, lengths):
        emb = self.embedding(X)
        packed_emb = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, (hidden, cell) = self.lstm(packed_emb)

        features = torch.cat([hidden[2, :, :], hidden[3, :, :]], dim=1)
        features = self.dropout(features)

        class_out = self.classifier(features)
        urgency_out = torch.sigmoid(self.regressor(features)).squeeze(1)

        return class_out, urgency_out
