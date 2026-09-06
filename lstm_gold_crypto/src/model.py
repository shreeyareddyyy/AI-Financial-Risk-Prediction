import torch
from torch import nn


class PriceVolatilityLSTM(nn.Module):
    """
    Multivariate LSTM with two prediction heads.

    Head 1:
        Next-day log return

    Head 2:
        Log of next 5-day future volatility
    """

    def __init__(
        self,
        input_size,
        hidden1=96,
        hidden2=48,
        dropout=0.2
    ):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden1,
            batch_first=True
        )

        self.drop1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(
            input_size=hidden1,
            hidden_size=hidden2,
            batch_first=True
        )

        self.drop2 = nn.Dropout(dropout)

        self.shared = nn.Sequential(
            nn.Linear(hidden2, 32),
            nn.ReLU()
        )

        # Next-day log-return head
        self.return_head = nn.Linear(32, 1)

        # Log-volatility head.
        #
        # IMPORTANT:
        # Do NOT use Softplus here.
        # The target is standardized log-volatility and
        # therefore can be negative.
        self.volatility_head = nn.Linear(32, 1)

    def forward(self, x):

        x, _ = self.lstm1(x)
        x = self.drop1(x)

        x, _ = self.lstm2(x)
        x = self.drop2(x[:, -1, :])

        shared = self.shared(x)

        predicted_return = self.return_head(shared)
        predicted_log_volatility = self.volatility_head(shared)

        return predicted_return, predicted_log_volatility