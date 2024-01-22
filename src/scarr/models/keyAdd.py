# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

import numpy as np
from .model import Model
from .utils import WEIGHTS, KEYS

class KeyAdd(Model):

    def __init__(self) -> None:
        self.num_vals = 9
        self.vals = np.arange(9)

    def calculate(self, plaintext: np.ndarray, key):
        inputs = np.bitwise_xor(plaintext, key)

        return WEIGHTS[inputs]
    
    def calculate_table(self, plaintext: np.ndarray):
        inputs = np.bitwise_xor(plaintext, KEYS)

        return WEIGHTS[inputs]
    
    def calculate_profile(self, batch):

        inputs = np.bitwise_xor(batch[0], batch[1])

        return WEIGHTS[inputs]