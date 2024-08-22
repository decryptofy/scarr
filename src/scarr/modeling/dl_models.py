# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

import torch
import torch.nn as nn


class DL_Models:

    # Multi-Layered Perceptron
    class MLP(nn.Module):
        def __init__(self, samples_len):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(samples_len, 120),
                nn.ReLU(),
                nn.BatchNorm1d(120),
                nn.Linear(120, 90),
                nn.ReLU(),
                nn.BatchNorm1d(90),
                nn.Linear(90, 50),
                nn.ReLU(),
                nn.BatchNorm1d(50),
                nn.Linear(50, 2),
                nn.Softmax(dim=1)
            )
        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits


    # Convolutional Neural Network
    class CNN(nn.Module):
        def __init__(self, samples_len):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Unflatten(1, (1, samples_len)),
                nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=4),
                nn.Flatten(),
                nn.Linear(16 * ((samples_len - 2) // 4), 2),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits
