# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.


import numpy as np
from .guess_model_value import GuessModelValue


class ModelBits(GuessModelValue):

    def __init__(self, model, bias=True) -> None:
        self.num_vals = 2
        self.vals = np.arange(2)
        self.num_bits = (model.num_vals-1).bit_length() * len(model) + (1 if bias else 0)
        self.model = model
        self.bias = bias

    def calculate(self, batch):
        outputs = self.model.calculate(batch)

        unpacked = np.unpackbits(outputs[..., np.newaxis], axis=-1, bitorder='little')

        if self.bias:
            bias_bits = np.ones(unpacked.shape[:-1]+(1,), dtype='uint8')
            unpacked = np.concatenate((unpacked, bias_bits), axis=-1)

        return unpacked

    def calculate_table(self, batch):
        outputs = self.model.calculate_table(batch)

        unpacked = np.unpackbits(np.atleast_3d(outputs), axis=-1, bitorder='little')

        if self.bias:
            bias_bits = np.ones(unpacked.shape[:-1]+(1,), dtype='uint8')
            unpacked = np.concatenate((unpacked, bias_bits), axis=-1)

        return unpacked

    def calculate_all_tables(self, batch):
        outputs = self.model.calculate_all_tables(batch)

        unpacked = np.unpackbits(outputs, axis=-2, bitorder='little')
        if self.bias:
            bias_shape = unpacked.shape[:-2] + (1,) + unpacked.shape[-1:]
            bias_bits = np.ones(bias_shape, dtype=unpacked.dtype)
            unpacked = np.concatenate((unpacked, bias_bits), axis=-2)

        return unpacked
