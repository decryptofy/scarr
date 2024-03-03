# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

import numpy as np
from .guess_model_value import GuessModelValue
from .utils import AES_SBOX, WEIGHTS, KEYS


class SboxDistance(GuessModelValue):

    def __init__(self) -> None:
        self.num_vals = 9
        self.vals = np.arange(9)

    def calculate(self, batch):
        inputs = np.bitwise_xor(batch[0], batch[1])

        outputs = AES_SBOX[inputs]

        return WEIGHTS[np.bitwise_xor(inputs, outputs)]

    def calculate_table(self, batch):
        inputs = np.bitwise_xor(np.squeeze(batch[0]), KEYS)

        outputs = AES_SBOX[inputs]

        return WEIGHTS[np.bitwise_xor(inputs, outputs)]

    def calculate_all_tables(self, batch):
        return np.apply_along_axis(self.calculate_all_tables_helper, axis=1, arr=batch[0])

    def calculate_all_tables_helper(self, data):
        inputs = np.bitwise_xor(data, KEYS)

        outputs = AES_SBOX[inputs]

        return WEIGHTS[np.bitwise_xor(inputs, outputs)]
