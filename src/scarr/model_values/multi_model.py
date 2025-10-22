# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.


import numpy as np
from .model_value import ModelValue
from .guess_model_value import GuessModelValue


class MultiModel(GuessModelValue):

    def __init__(self, models) -> None:
        self.models = models
        # Expects that all models produce same val range
        self.num_vals = models[0].num_vals
        self.vals = models[0].vals

    def __len__(self):
        return len(self.models)

    def calculate(self, batch):
        all_models_output = self.models[0].calculate(batch)[np.newaxis, ...]
        for m in self.models[1:]:
            new_output = m.calculate(batch)[np.newaxis, ...]
            all_models_output = np.concatenate((all_models_output), axis=0)

        return all_models_output

    def calculate_table(self, batch):
        all_models_output = self.models[0].calculate_table(batch)[..., np.newaxis] 
        for m in self.models[1:]:
            new_output = m.calculate_table(batch)[..., np.newaxis]
            all_models_output = np.concatenate((all_models_output, new_output), axis=-1)

        return all_models_output

    def calculate_all_tables(self, batch):
        all_models_output = self.models[0].calculate_all_tables(batch)[..., np.newaxis]
        for m in self.models[1:]:
            new_output = m.calculate_all_tables(batch)[..., np.newaxis]
            all_models_output = np.concatenate((all_models_output, new_output), axis=-1)

        return all_models_output.swapaxes(-1, -2)
