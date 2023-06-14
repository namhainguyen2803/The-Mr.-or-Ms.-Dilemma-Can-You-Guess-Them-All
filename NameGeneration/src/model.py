"""
    Created by @namhainguyen2803 in 10/05/2023
"""
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNN(nn.Module):
    def __init__(self, dictionary, embedding_dim=15, num_lstm_units=15, num_lstm_layers=3, device=DEVICE):
        super().__init__()
        self.vocab_size = len(dictionary)

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=embedding_dim
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=num_lstm_units,
            num_layers=num_lstm_layers,
            batch_first=True
        )

        self.linear_hidden_to_output = nn.Linear(num_lstm_units, self.vocab_size)

        self.dictionary = dictionary
        self.device = device
        self.to(device)

    def forward(self, input, hidden=None):
        embeddings = self.embedding(input)
        if hidden is None:
            out, (h, c) = self.lstm(embeddings)
        else:
            out, (h, c) = self.lstm(embeddings, hidden)

        out = out.contiguous().view(-1, out.shape[-1])
        logits = self.linear_hidden_to_output(out)
        return logits, (h.detach(), c.detach())

    def sample(self):
        with torch.no_grad():
            self.eval()
            h_prev = None
            texts = []
            # x = np.random.choice(np.array([14]), 1)[None, :]
            x = np.random.choice(np.array([20, 4, 21, 13, 24, 2, 12, 3, 11, 16, 7, 22, 17, 15, 14, 19, 8, 5, 25]), 1)[None, :]
            texts.append(x[0][0])
            x = torch.from_numpy(x).type(torch.int64).to(self.device)
            i = 0
            while True:
                i += 1
                logits, h_prev = self.forward(x, h_prev)
                np_logits = logits.detach().cpu().numpy()
                prob = np.exp(np_logits) / np.sum(np.exp(np_logits), axis=1)
                id = np.random.choice(np.arange(self.vocab_size), p=prob.ravel())
                texts.append(id)
                if id == 0 or i == 25:
                    break
                else:
                    x = torch.tensor(np.expand_dims(id, axis=(0, 1)), dtype=torch.int64).to(self.device)
        return texts
